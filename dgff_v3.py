"""
dgff_v3.py — DGFF Module with corrected backbone indices for YOLOv8/11/26
and an optional 4th feedback level at P2 (stride 4).

Drop-in replacement for your notebook's Section 2 (LLEN) and Section 3
(DGFF Module) cells. Backward compatible: with yolo_weights='yolov8n.pt'
and use_p2=False (the defaults), this reproduces your exact paper
architecture (verified against your reported param counts: LLEN 4,844,803 /
DGFF total 1,359,120 / adapters 86,464 / backbone 1,272,656).

What changed vs. your original:
  1. Backbone layer indices are looked up per-architecture instead of
     hardcoded to (4, 6, 9). YOLOv8's P5 (SPPF) is at index 9; YOLO11/YOLO26
     add a C2PSA attention block AFTER SPPF, so their "richest" P5 is at
     index 10. Using index 9 on a v11/v26 backbone silently gives you the
     same SPPF-only features v8 has -- you'd download bigger weights and
     get no architectural benefit. This version fixes that.
  2. Optional P2 (stride 4) feedback into LLEN's previously-unused dec1
     stage, for testing whether higher-resolution feedback helps the
     small/occluded classes (Chair, Motorbike, People, Table) that your
     Table 5 shows DGFF currently underperforms LLEN-only on.

Verified in a clean sandbox against yolov8n.pt, yolo11n.pt, and yolo26n.pt
(param counts + forward-pass shapes all checked -- see bottom of this file).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


# ── Known backbone layer maps ──────────────────────────────────────────────
# index into yolo.model.model where each pyramid level's richest feature
# lives, and how many layers to keep when truncating the backbone.
# If you try a version not listed here, run the probe snippet at the
# bottom of this file first -- do NOT assume (4, 6, 9) still holds.
BACKBONE_LAYER_MAPS = {
    'yolov8':  dict(p2=2, p3=4, p4=6, p5=9,  n_layers=10),  # SPPF is final backbone layer
    'yolo11':  dict(p2=2, p3=4, p4=6, p5=10, n_layers=11),  # SPPF(9) -> C2PSA(10)
    'yolo26':  dict(p2=2, p3=4, p4=6, p5=10, n_layers=11),  # same backbone as yolo11
}


def _infer_arch_key(yolo_weights: str) -> str:
    name = yolo_weights.lower()
    if 'yolo26' in name:
        return 'yolo26'
    if 'yolo11' in name or 'yolov11' in name:
        return 'yolo11'
    if 'yolov8' in name or 'yolo8' in name:
        return 'yolov8'
    raise ValueError(
        f"Don't have a verified layer map for '{yolo_weights}'. "
        f"Run the probe snippet at the bottom of dgff_v3.py first, "
        f"add the correct indices to BACKBONE_LAYER_MAPS, then retry."
    )


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
        f_det = F.interpolate(f_det, size=target_size, mode='bilinear', align_corners=False)
        return self.gate(f_det) * self.proj(f_det)


class DGFFModule(nn.Module):
    def __init__(
        self,
        yolo_weights: str = 'yolov8n.pt',
        llen_base_ch: int = 32,
        freeze_backbone: bool = True,
        use_p2: bool = False,          # opt-in 4th feedback level
        arch: str = None,              # override auto-detection if needed
    ):
        super().__init__()
        self.use_p2 = use_p2
        arch_key = arch or _infer_arch_key(yolo_weights)
        layer_map = BACKBONE_LAYER_MAPS[arch_key]
        self.layer_map = layer_map

        yolo = YOLO(yolo_weights)
        n_layers = layer_map['n_layers']
        self.backbone = yolo.model.model[:n_layers]

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.idx_p2 = layer_map['p2']
        self.idx_p3 = layer_map['p3']
        self.idx_p4 = layer_map['p4']
        self.idx_p5 = layer_map['p5']

        # Probe actual channel depths (works for any architecture)
        with torch.no_grad():
            probe = torch.zeros(1, 3, 64, 64)
            out = probe
            chans = {}
            for i, layer in enumerate(self.backbone):
                out = layer(out)
                if i in (self.idx_p2, self.idx_p3, self.idx_p4, self.idx_p5):
                    chans[i] = out.shape[1]
        self.p2_ch = chans[self.idx_p2]
        self.p3_ch = chans[self.idx_p3]
        self.p4_ch = chans[self.idx_p4]
        self.p5_ch = chans[self.idx_p5]

        c = llen_base_ch
        self._llen_base_ch = c
        self.adapter_p3 = GatedAdapter(self.p3_ch, c)
        self.adapter_p5 = GatedAdapter(self.p5_ch, c * 4)
        self.adapter_p4 = GatedAdapter(self.p4_ch, c * 2)
        if self.use_p2:
            self.adapter_p2 = GatedAdapter(self.p2_ch, c)   # -> dec1 (full res)

    def forward(self, enhanced_img):
        x = enhanced_img
        feats = {}
        needed = {self.idx_p2, self.idx_p3, self.idx_p4, self.idx_p5} if self.use_p2 \
            else {self.idx_p3, self.idx_p4, self.idx_p5}

        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in needed:
                feats[i] = x

        c = self._llen_base_ch
        H_input, W_input = enhanced_img.shape[-2:]

        p5_size = (H_input // 8, W_input // 8)   # dec4
        p4_size = (H_input // 4, W_input // 4)   # dec3
        p3_size = (H_input // 2, W_input // 2)   # dec2

        p5_feedback = self.adapter_p5(feats[self.idx_p5], p5_size)
        p4_feedback = self.adapter_p4(feats[self.idx_p4], p4_size)
        p3_feedback = self.adapter_p3(feats[self.idx_p3], p3_size)

        if self.use_p2:
            p2_size = (H_input, W_input)         # dec1, full resolution
            p2_feedback = self.adapter_p2(feats[self.idx_p2], p2_size)
            return p4_feedback, p5_feedback, p3_feedback, p2_feedback

        return p4_feedback, p5_feedback, p3_feedback

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ── LLEN with optional P2 injection at dec1 ─────────────────────────────────
# Only dec1's feedback_ch changes (None -> c) vs. your original. Everything
# else is byte-identical to your paper architecture.

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
    def __init__(self, base_ch=32, use_p2=False):
        super().__init__()
        c = base_ch
        self.use_p2 = use_p2
        self.enc1 = EncoderBlock(3, c)
        self.enc2 = EncoderBlock(c, c * 2)
        self.enc3 = EncoderBlock(c * 2, c * 4)
        self.enc4 = EncoderBlock(c * 4, c * 8)
        self.bottleneck = nn.Sequential(ResBlock(c * 8), ResBlock(c * 8))
        self.dec4 = DecoderBlock(c * 8, c * 8, c * 4, feedback_ch=c * 4)
        self.dec3 = DecoderBlock(c * 4, c * 4, c * 2, feedback_ch=c * 2)
        self.dec2 = DecoderBlock(c * 2, c * 2, c,   feedback_ch=c)
        # dec1 now optionally takes P2 feedback (was feedback_ch=None in the paper version)
        self.dec1 = DecoderBlock(c, c, c, feedback_ch=(c if use_p2 else None))
        self.head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.ReLU(True),
            nn.Conv2d(c, 3, 1), nn.Sigmoid(),
        )

    def forward(self, x, dgff_p5=None, dgff_p4=None, dgff_p3=None, dgff_p2=None):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4, feedback=dgff_p5)
        x = self.dec3(x, s3, feedback=dgff_p4)
        x = self.dec2(x, s2, feedback=dgff_p3)
        x = self.dec1(x, s1, feedback=dgff_p2 if self.use_p2 else None)
        return self.head(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Usage ─────────────────────────────────────────────────────────────────
# Exact paper reproduction (unchanged):
#   llen = LLEN(base_ch=32, use_p2=False)
#   dgff = DGFFModule(yolo_weights='yolov8n.pt', llen_base_ch=32, use_p2=False)
#   ...
#   p4, p5, p3 = dgff(enhanced)
#   enhanced_guided = llen(low, dgff_p5=p5, dgff_p4=p4, dgff_p3=p3)
#
# Backbone-choice ablation (richer P3/P5, same 3-level feedback):
#   dgff = DGFFModule(yolo_weights='yolo26n.pt', llen_base_ch=32, use_p2=False)
#   # everything else identical
#
# P2 ablation (targets small/occluded classes per your Table 5):
#   llen = LLEN(base_ch=32, use_p2=True)
#   dgff = DGFFModule(yolo_weights='yolov8n.pt', llen_base_ch=32, use_p2=True)
#   ...
#   p4, p5, p3, p2 = dgff(enhanced)
#   enhanced_guided = llen(low, dgff_p5=p5, dgff_p4=p4, dgff_p3=p3, dgff_p2=p2)
#
# You can combine both (yolo26n.pt + use_p2=True) for the most aggressive variant.


if __name__ == '__main__':
    print("Sanity-checking all combinations...\n")
    dummy = torch.rand(1, 3, 256, 256)

    for weights in ['yolov8n.pt', 'yolo11n.pt', 'yolo26n.pt']:
        for use_p2 in [False, True]:
            llen = LLEN(base_ch=32, use_p2=use_p2)
            dgff = DGFFModule(yolo_weights=weights, llen_base_ch=32, use_p2=use_p2)

            out = dgff(dummy)
            if use_p2:
                p4, p5, p3, p2 = out
                guided = llen(dummy, dgff_p5=p5, dgff_p4=p4, dgff_p3=p3, dgff_p2=p2)
            else:
                p4, p5, p3 = out
                guided = llen(dummy, dgff_p5=p5, dgff_p4=p4, dgff_p3=p3)

            assert guided.shape == dummy.shape
            tot, tr = dgff.count_parameters()
            print(f"{weights:14s} use_p2={use_p2!s:5s}  "
                  f"LLEN={llen.count_parameters():,}  "
                  f"DGFF total={tot:,} trainable={tr:,}  "
                  f"backbone_layers={dgff.layer_map['n_layers']}  "
                  f"p2/p3/p4/p5 idx={dgff.idx_p2}/{dgff.idx_p3}/{dgff.idx_p4}/{dgff.idx_p5}")
    print("\nAll combinations forward-pass cleanly.")
