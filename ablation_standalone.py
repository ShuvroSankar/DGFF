"""
ablation_standalone.py — real lambda_det ablation, runs from a plain
terminal, no Jupyter required. Built directly from your notebook's actual
cells (LOLDataset, LLEN, DGFFModule, PerceptualLoss, DetectionFeatureLoss,
get_lambda_det, validate) -- not reconstructed from scratch.

Resumable at the CONFIG level: if interrupted, completed lambda_det
configs are kept (results appended to ablation_results.json after each
one finishes); only the in-progress config's partial epochs are lost.

Recommended way to run so it survives you closing the terminal/browser:

    nohup caffeinate -i python3 ablation_standalone.py > ablation_log.txt 2>&1 &

Then check progress any time with:  tail -f ablation_log.txt
"""

import os
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models

from PIL import Image
import numpy as np

from ultralytics import YOLO

# ── Device ───────────────────────────────────────────────────────────────
DEVICE = (
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cuda') if torch.cuda.is_available()         else
    torch.device('cpu')
)
print(f"Device: {DEVICE}")

CFG = dict(
    lol_root      = './LOL',
    crop_size     = 256,
    base_ch       = 32,
    yolo_weights  = 'yolov8n.pt',
    batch_size    = 4,
    lr            = 1e-4,
    lambda_perc   = 0.1,
    num_workers   = 0,
    ckpt_dir      = './checkpoints_ablation',
)
Path(CFG['ckpt_dir']).mkdir(parents=True, exist_ok=True)

ABLATION_LAMBDAS = [0.0, 0.1, 0.5, 1.0]
ABLATION_EPOCHS = 50
ABLATION_SEED = 42
RESULTS_PATH = 'ablation_results.json'


# ── Dataset (exact copy from your notebook, Cell 4) ────────────────────────

class LOLDataset(Dataset):
    SPLIT_DIRS = {'train': 'our485', 'val': 'eval15'}
    _IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(self, root, split='train', crop_size=256, augment=True):
        assert split in self.SPLIT_DIRS
        self.crop_size = crop_size
        self.augment = augment and (split == 'train')

        base = Path(root) / self.SPLIT_DIRS[split]
        self.low_dir = base / 'low'
        self.high_dir = base / 'high'

        for d in (self.low_dir, self.high_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"Not found: {d}")

        self.filenames = sorted(
            f for f in os.listdir(self.low_dir)
            if Path(f).suffix.lower() in self._IMG_EXTS
        )
        missing = [f for f in self.filenames if not (self.high_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing high images: {missing[:3]}")

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        low_img = Image.open(self.low_dir / fname).convert('RGB')
        high_img = Image.open(self.high_dir / fname).convert('RGB')

        if self.crop_size is not None:
            low_img, high_img = self._paired_crop(low_img, high_img)

        if self.augment and random.random() > 0.5:
            low_img = TF.hflip(low_img)
            high_img = TF.hflip(high_img)

        return self.to_tensor(low_img), self.to_tensor(high_img), fname

    def _paired_crop(self, low, high):
        w, h = low.size
        c = self.crop_size
        if w < c or h < c:
            return TF.resize(low, [c, c]), TF.resize(high, [c, c])
        top = random.randint(0, h - c)
        left = random.randint(0, w - c)
        return TF.crop(low, top, left, c, c), TF.crop(high, top, left, c, c)


def get_loaders(cfg):
    train_set = LOLDataset(cfg['lol_root'], 'train', cfg['crop_size'], augment=True)
    val_set = LOLDataset(cfg['lol_root'], 'val', None, augment=False)
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'],
                               shuffle=True, num_workers=cfg['num_workers'],
                               drop_last=True, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=1,
                             shuffle=False, num_workers=cfg['num_workers'],
                             drop_last=False, pin_memory=False)
    return train_loader, val_loader


# ── LLEN (exact copy from Cell 6) ───────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )
        self.res = ResBlock(out_ch)
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.res(self.conv(x))
        skip = x
        return self.down(x), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, feedback_ch=32):
        super().__init__()
        self.merge = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )
        self.res = ResBlock(out_ch)
        if feedback_ch is not None:
            self.feedback_proj = nn.Conv2d(feedback_ch, out_ch, 1, bias=False)
        else:
            self.feedback_proj = None

    def forward(self, x, skip, feedback=None):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = self.merge(torch.cat([x, skip], dim=1))
        if feedback is not None and self.feedback_proj is not None:
            if feedback.shape[-2:] != x.shape[-2:]:
                feedback = F.interpolate(feedback, size=x.shape[-2:],
                                          mode='bilinear', align_corners=False)
            x = x + self.feedback_proj(feedback)
        return self.res(x)


class LLEN(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        c = base_ch
        self.enc1 = EncoderBlock(3, c)
        self.enc2 = EncoderBlock(c, c * 2)
        self.enc3 = EncoderBlock(c * 2, c * 4)
        self.enc4 = EncoderBlock(c * 4, c * 8)
        self.bottleneck = nn.Sequential(ResBlock(c * 8), ResBlock(c * 8))
        self.dec4 = DecoderBlock(c * 8, c * 8, c * 4, feedback_ch=c * 4)
        self.dec3 = DecoderBlock(c * 4, c * 4, c * 2, feedback_ch=c * 2)
        self.dec2 = DecoderBlock(c * 2, c * 2, c, feedback_ch=c)
        self.dec1 = DecoderBlock(c, c, c, feedback_ch=None)
        self.head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.ReLU(True),
            nn.Conv2d(c, 3, 1), nn.Sigmoid(),
        )

    def forward(self, x, dgff_p5=None, dgff_p4=None, dgff_p3=None):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4, feedback=dgff_p5)
        x = self.dec3(x, s3, feedback=dgff_p4)
        x = self.dec2(x, s2, feedback=dgff_p3)
        x = self.dec1(x, s1)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── DGFF Module (exact copy from Cell 8) ────────────────────────────────────

class GatedAdapter(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, f_det, target_size):
        f_det = nn.functional.interpolate(
            f_det, size=target_size, mode='bilinear', align_corners=False
        )
        return self.gate(f_det) * self.proj(f_det)


class DGFFModule(nn.Module):
    def __init__(self, yolo_weights='yolov8n.pt', llen_base_ch=32, freeze_backbone=True):
        super().__init__()
        yolo = YOLO(yolo_weights)
        self.backbone = yolo.model.model[:10]

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        with torch.no_grad():
            probe = torch.zeros(1, 3, 64, 64)
            out = probe
            for i, layer in enumerate(self.backbone):
                out = layer(out)
                if i == 4:
                    actual_p3 = out.shape[1]
                if i == 6:
                    actual_p4 = out.shape[1]
                if i == 9:
                    actual_p5 = out.shape[1]
        self.p3_ch = actual_p3
        self.p4_ch = actual_p4
        self.p5_ch = actual_p5

        c = llen_base_ch
        self._llen_base_ch = c
        self.adapter_p3 = GatedAdapter(self.p3_ch, c)
        self.adapter_p5 = GatedAdapter(self.p5_ch, c * 4)
        self.adapter_p4 = GatedAdapter(self.p4_ch, c * 2)

    def forward(self, enhanced_img):
        x = enhanced_img
        p4_raw = p5_raw = p3_raw = None
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 4:
                p3_raw = x
            if i == 6:
                p4_raw = x
            if i == 9:
                p5_raw = x

        c = self._llen_base_ch
        H_input, W_input = enhanced_img.shape[-2:]
        p5_size = (H_input // 8, W_input // 8)
        p4_size = (H_input // 4, W_input // 4)
        p3_size = (H_input // 2, W_input // 2)

        p5_feedback = self.adapter_p5(p5_raw, p5_size)
        p4_feedback = self.adapter_p4(p4_raw, p4_size)
        p3_feedback = self.adapter_p3(p3_raw, p3_size)
        return p4_feedback, p5_feedback, p3_feedback

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ── Losses / metrics (exact copy from Cell 10) ──────────────────────────────

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg)[:10]).to(device).eval()
        self.slice2 = nn.Sequential(*list(vgg)[:17]).to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        return F.mse_loss(self.slice1(pred), self.slice1(target)) + \
               F.mse_loss(self.slice2(pred), self.slice2(target))


class DetectionFeatureLoss(nn.Module):
    def __init__(self, dgff_module):
        super().__init__()
        self.backbone = dgff_module.backbone

    def _extract(self, x, no_grad=False):
        feats = []
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            for i, layer in enumerate(self.backbone):
                x = layer(x)
                if i in (4, 6, 9):
                    feats.append(x)
        return feats

    def forward(self, enhanced_guided, high):
        feats_enh = self._extract(enhanced_guided, no_grad=False)
        feats_clean = self._extract(high, no_grad=True)
        return sum(F.mse_loss(fe, fc.detach()) for fe, fc in zip(feats_enh, feats_clean))


def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    return float('inf') if mse == 0 else 10 * torch.log10(torch.tensor(1.0 / mse)).item()


def compute_ssim(pred, target):
    try:
        from pytorch_msssim import ssim
        return ssim(pred, target, data_range=1.0).item()
    except ImportError:
        mu1, mu2 = pred.mean(), target.mean()
        s1, s2 = pred.std(), target.std()
        s12 = ((pred - mu1) * (target - mu2)).mean()
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) /
                ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 ** 2 + s2 ** 2 + C2))).item()


def get_lambda_det(epoch, total_epochs, max_lambda):
    warmup = total_epochs // 2
    return max_lambda * min(1.0, epoch / warmup)


@torch.no_grad()
def validate(llen, dgff, val_loader, device):
    llen.eval()
    dgff.eval()
    total_psnr = total_ssim = 0.0
    for low, high, _ in val_loader:
        low, high = low.to(device), high.to(device)
        enhanced = llen(low)
        p4, p5, p3 = dgff(enhanced)
        enhanced_guided = llen(low, dgff_p5=p5, dgff_p4=p4, dgff_p3=p3)
        total_psnr += compute_psnr(enhanced_guided, high)
        total_ssim += compute_ssim(enhanced_guided, high)
    n = len(val_loader)
    return total_psnr / n, total_ssim / n


# ── Ablation loop ────────────────────────────────────────────────────────

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_result(result):
    results = load_results()
    results.append(result)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)


def run_config(lambda_max, epochs, seed, train_loader, val_loader):
    torch.manual_seed(seed)

    llen = LLEN(base_ch=CFG['base_ch']).to(DEVICE)
    dgff = DGFFModule(yolo_weights=CFG['yolo_weights'], llen_base_ch=CFG['base_ch'],
                       freeze_backbone=True).to(DEVICE)
    perc_loss = PerceptualLoss(DEVICE)
    det_loss = DetectionFeatureLoss(dgff).to(DEVICE)
    l1_loss = nn.L1Loss()

    trainable = (list(llen.parameters()) + list(dgff.adapter_p3.parameters()) +
                 list(dgff.adapter_p4.parameters()) + list(dgff.adapter_p5.parameters()))
    optimiser = Adam(trainable, lr=CFG['lr'], betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)

    best_psnr, best_ssim = 0.0, 0.0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        llen.train()
        dgff.train()
        lam_det = get_lambda_det(epoch, epochs, lambda_max)
        ep_loss = 0.0

        for low, high, _ in train_loader:
            low, high = low.to(DEVICE), high.to(DEVICE)
            enhanced = llen(low)
            p4, p5, p3 = dgff(enhanced)
            enhanced_guided = llen(low, dgff_p5=p5, dgff_p4=p4, dgff_p3=p3)

            loss_l1 = l1_loss(enhanced_guided, high)
            loss_perc = perc_loss(enhanced_guided, high)
            loss_det = det_loss(enhanced_guided, high) if lam_det > 0 else torch.tensor(0.0, device=DEVICE)
            loss = loss_l1 + CFG['lambda_perc'] * loss_perc + lam_det * loss_det

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimiser.step()
            ep_loss += loss.item()

        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            psnr, ssim = validate(llen, dgff, val_loader, DEVICE)
            best_psnr, best_ssim = max(best_psnr, psnr), max(best_ssim, ssim)
            elapsed = time.time() - t_start
            print(f"  [lambda_det={lambda_max}] epoch {epoch:03d}/{epochs}  "
                  f"loss={ep_loss/len(train_loader):.4f}  PSNR={psnr:.2f}dB  "
                  f"SSIM={ssim:.4f}  ({elapsed/60:.1f} min elapsed)", flush=True)

    torch.save({'llen_state': llen.state_dict(), 'dgff_state': dgff.state_dict(),
                'best_psnr': best_psnr, 'best_ssim': best_ssim},
               Path(CFG['ckpt_dir']) / f'lambda_{lambda_max}.pt')

    return {'lambda_det': lambda_max, 'epochs': epochs, 'seed': seed,
            'best_psnr': best_psnr, 'best_ssim': best_ssim,
            'minutes': (time.time() - t_start) / 60}


def main():
    print(f"Loading LOL dataset from {CFG['lol_root']}...")
    train_loader, val_loader = get_loaders(CFG)
    print(f"Train: {len(train_loader.dataset)} pairs | Val: {len(val_loader.dataset)} pairs")

    done_lambdas = {r['lambda_det'] for r in load_results()}
    print(f"\nAlready completed: {sorted(done_lambdas) or 'none'}")

    for lam in ABLATION_LAMBDAS:
        if lam in done_lambdas:
            print(f"\nSkipping lambda_det={lam} (already in {RESULTS_PATH})")
            continue
        print(f"\n=== lambda_det = {lam} ===", flush=True)
        result = run_config(lam, ABLATION_EPOCHS, ABLATION_SEED, train_loader, val_loader)
        save_result(result)
        print(f"  Done: PSNR={result['best_psnr']:.2f}dB SSIM={result['best_ssim']:.4f} "
              f"({result['minutes']:.1f} min) -- saved to {RESULTS_PATH}")

    print("\n=== Final ablation table ===")
    results = sorted(load_results(), key=lambda r: r['lambda_det'])
    print(f"{'lambda_det':<12}{'PSNR (dB)':<12}{'SSIM':<10}")
    for r in results:
        print(f"{r['lambda_det']:<12}{r['best_psnr']:<12.2f}{r['best_ssim']:<10.4f}")


if __name__ == '__main__':
    main()
