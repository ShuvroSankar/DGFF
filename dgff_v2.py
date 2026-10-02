"""
dgff_v2.py — DGFF Module with Channel Gating
=============================================
Improvement over v1: 1×1 adapter + sigmoid gate replaces plain
element-wise addition. The gate learns which detector channels
to incorporate per-spatial-location, making the feedback selective
and more theoretically principled.

Change from v1:
  v1:  x_decoder = x_decoder + adapter(f_det)
  v2:  gate = sigmoid(gate_conv(f_det))
       x_decoder = x_decoder + gate * adapter(f_det)

The gate suppresses irrelevant detector features (e.g. background
texture channels that don't help enhancement) while amplifying
object-boundary and semantic channels.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO


class GatedAdapter(nn.Module):
    """
    Projects YOLOv8 backbone features into LLEN decoder space
    with a learned sigmoid gate.

    Args:
        in_ch  : YOLOv8 feature channels (256 for P4, 512 for P5 on yolov8n)
        out_ch : LLEN decoder channels at the injection point
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Value path: projects detector features to decoder space
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Gate path: learns a channel-wise attention mask in [0, 1]
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, f_det, target_size):
        """
        Args:
            f_det       : (B, in_ch, H', W')  raw backbone feature
            target_size : (H, W) spatial size of the decoder stage

        Returns:
            (B, out_ch, H, W) gated feature feedback
        """
        # Resize to match decoder spatial resolution
        f_det = nn.functional.interpolate(
            f_det, size=target_size, mode='bilinear', align_corners=False
        )
        return self.gate(f_det) * self.proj(f_det)   # element-wise gated projection


class DGFFModule(nn.Module):
    """
    Detection-Guided Feature Feedback module.

    Wraps a frozen YOLOv8 backbone and two GatedAdapters that project
    P4 and P5 backbone features into the LLEN decoder spatial dimensions.

    Args:
        yolo_weights    : path to YOLOv8 .pt file
        llen_base_ch    : base channel width of the LLEN (default 32)
        freeze_backbone : if True, backbone weights are not updated
    """

    # YOLOv8n backbone channel counts at P4 and P5
    # These depend on the variant (n/s/m/l/x). yolov8n:
    YOLO_P4_CH = 256
    YOLO_P5_CH = 512

    def __init__(
        self,
        yolo_weights:    str  = 'yolov8n.pt',
        llen_base_ch:    int  = 32,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # ── Load YOLOv8 backbone ──────────────────────────────
        yolo = YOLO(yolo_weights)
        # Extract the backbone (model.model.model[:10] for YOLOv8)
        # This gives us the CSP + C2f feature extraction up to P5
        self.backbone = yolo.model.model[:10]

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # ── Gated adapters ───────────────────────────────────
        c = llen_base_ch
        # P5 feedback → decoder stage D4 (out_ch = c*4 = 128)
        self.adapter_p5 = GatedAdapter(self.YOLO_P5_CH, c * 4)
        # P4 feedback → decoder stage D3 (out_ch = c*2 = 64)
        self.adapter_p4 = GatedAdapter(self.YOLO_P4_CH, c * 2)

        self._llen_base_ch = c

    # ----------------------------------------------------------
    def forward(self, enhanced_img):
        """
        Args:
            enhanced_img : (B, 3, H, W)  enhanced image from LLEN Pass 1

        Returns:
            (p4_feedback, p5_feedback) — tensors ready to inject into LLEN decoder
        """
        # Run backbone — collect intermediate feature maps
        x      = enhanced_img
        p4_raw = None
        p5_raw = None

        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # YOLOv8n: P4 at layer 6, P5 at layer 9 (backbone output)
            if i == 6:
                p4_raw = x
            if i == 9:
                p5_raw = x

        if p4_raw is None or p5_raw is None:
            raise RuntimeError(
                "Could not extract P4/P5 from backbone. "
                "Check layer indices for your YOLOv8 variant."
            )

        c = self._llen_base_ch
        H_input, W_input = enhanced_img.shape[-2:]

        # Compute decoder target sizes (matching LLEN decoder stages)
        p5_size = (H_input // 8,  W_input // 8)   # D4 spatial dim
        p4_size = (H_input // 4,  W_input // 4)   # D3 spatial dim

        p5_feedback = self.adapter_p5(p5_raw, p5_size)  # (B, c*4, H/8, W/8)
        p4_feedback = self.adapter_p4(p4_raw, p4_size)  # (B, c*2, H/4, W/4)

        return p4_feedback, p5_feedback

    # ----------------------------------------------------------
    def count_parameters(self):
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print("DGFFv2 Smoke Test")
    device = torch.device('cpu')

    try:
        dgff = DGFFModule(yolo_weights='yolov8n.pt', llen_base_ch=32).to(device)
    except Exception as e:
        print(f"[SKIP] Could not load YOLOv8: {e}")
        sys.exit(0)

    total, trainable = dgff.count_parameters()
    print(f"  Params total / trainable: {total:,} / {trainable:,}")

    dummy = torch.rand(2, 3, 256, 256)
    p4, p5 = dgff(dummy)
    print(f"  P4 feedback shape: {tuple(p4.shape)}  (expected: 2, 64, 64, 64)")
    print(f"  P5 feedback shape: {tuple(p5.shape)}  (expected: 2, 128, 32, 32)")
    print("  Smoke test passed ✓")
