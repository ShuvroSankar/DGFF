import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


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
        p4_raw = None
        p5_raw = None
        p3_raw = None
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
