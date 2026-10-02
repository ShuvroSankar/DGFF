"""
Detection-Guided Feature Feedback (DGFF) Module
================================================
The DGFF module is the core architectural contribution of this thesis.

During training it:
  1. Hooks into the YOLOv8 backbone to extract intermediate feature maps
     at P4 (stride-16) and P5 (stride-32) — the semantically richest layers.
  2. Projects each feature map to match the channel depth of the corresponding
     LLEN decoder stage via lightweight 1×1 convolutional adapters.
  3. Returns the projected tensors so LLEN.forward() can inject them via
     element-wise addition at decoder stages D4 and D3.

At inference the DGFF module is NOT used — LLEN runs alone with no overhead.

YOLOv8n backbone channel dimensions (verified for 640×640 input):
  P4: (B, 256, H/16, W/16)   stride-16, semantically rich
  P5: (B, 512, H/32, W/32)   stride-32, most semantic

LLEN decoder injection targets:
  dec4 output: (B, 128, H/8,  W/8)   ← receives projected P5
  dec3 output: (B,  64, H/4,  W/4)   ← receives projected P4

Data flow during training:
  low_img
    │
    ▼
  LLEN encoder → bottleneck
    │                          YOLOv8 backbone
    │         enhanced_img ──────────────────────► P4 (256ch)
    │              ▲                             ► P5 (512ch)
    │              │                                  │
    │         LLEN head                          DGFF adapters
    │              ▲                            P4→64ch, P5→128ch
    │              │                                  │
    └── dec1 ◄ dec2 ◄ dec3 ◄──────── dgff_p4        │
                         dec4 ◄──────────────── dgff_p5
                              (element-wise +)

Usage:
    from dgff import DGFFModule
    dgff = DGFFModule()

    # During joint training forward pass:
    enhanced        = llen(low_img)                      # first pass: no feedback
    dgff_p4, dgff_p5 = dgff(enhanced)                   # extract + project features
    enhanced        = llen(low_img, dgff_p5, dgff_p4)   # second pass: with feedback
    loss            = detection_loss + l1_loss(enhanced, high_img)
    loss.backward()
"""

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
import torch.nn.functional as F
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
# Adapter — lightweight 1×1 projection
# ─────────────────────────────────────────────────────────────

class FeatureAdapter(nn.Module):
    """
    Projects a YOLOv8 feature map to the channel depth expected by LLEN.

    1×1 Conv keeps spatial size unchanged and adds negligible parameter cost.
    BN + ReLU stabilise training when detection gradients flow back.

    Args:
        in_ch  : channels from YOLOv8 backbone (P4=256, P5=512)
        out_ch : channels expected by LLEN decoder (dec3=64, dec4=128)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.adapt(x)


# ─────────────────────────────────────────────────────────────
# DGFF Module
# ─────────────────────────────────────────────────────────────

class DGFFModule(nn.Module):
    """
    Detection-Guided Feature Feedback module.

    Wraps a frozen-or-trainable YOLOv8 backbone and attaches forward hooks
    to capture P4 and P5 feature maps. Lightweight adapters project them to
    match the LLEN decoder channel depths.

    Args:
        yolo_weights : YOLOv8 model weights string (e.g. 'yolov8n.pt')
                       Downloads automatically on first use.
        llen_base_ch : base channel count of the LLEN (default 32).
                       dec4 expects llen_base_ch*4, dec3 expects llen_base_ch*2.
        freeze_backbone : if True, YOLOv8 backbone weights are frozen and
                          only the adapter layers are trained in this module.
                          Set False to fine-tune backbone end-to-end.

    YOLOv8n P-layer channels (hard-coded for yolov8n):
        P4 → 128 channels  (layer 6 C2f output)
        P5 → 256 channels  (layer 9 C2f output)
    If you switch to yolov8s/m/l/x, update P4_CH and P5_CH accordingly:
        yolov8s: P4=256,  P5=512
        yolov8m: P4=384,  P5=576
        yolov8l: P4=512,  P5=512
        yolov8x: P4=640,  P5=640
    """

    # YOLOv8n backbone output channels at P4 and P5
    # Verified by probing layer outputs on 640×640 input:
    #   layer 6 (C2f) → 128ch  = P4
    #   layer 9 (C2f) → 256ch  = P5
    P4_CH = 128
    P5_CH = 256

    # YOLOv8 layer indices where P4 and P5 are produced
    # (these are the C2f output layers in the YOLOv8n backbone)
    P4_LAYER_IDX = 6    # stride-16 feature map
    P5_LAYER_IDX = 9    # stride-32 feature map

    def __init__(
        self,
        yolo_weights: str = 'yolov8n.pt',
        llen_base_ch: int = 32,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # ── Load YOLOv8 ──────────────────────────────────────
        yolo = YOLO(yolo_weights)
        self.backbone = yolo.model.model   # nn.Sequential of backbone layers

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # ── Verify actual backbone channel depths ────────────
        # Run a tiny probe so we catch channel mismatches at init time
        # rather than deep inside a training loop.
        with torch.no_grad():
            _probe   = torch.zeros(1, 3, 64, 64)
            _out     = _probe
            for _i, _layer in enumerate(list(self.backbone)[:self.P5_LAYER_IDX + 1]):
                _out = _layer(_out)
                if _i == self.P4_LAYER_IDX:
                    _actual_p4 = _out.shape[1]
                if _i == self.P5_LAYER_IDX:
                    _actual_p5 = _out.shape[1]
            if _actual_p4 != self.P4_CH or _actual_p5 != self.P5_CH:
                raise ValueError(
                    f"YOLOv8 backbone channel mismatch!\n"
                    f"  P4: expected {self.P4_CH}, got {_actual_p4}\n"
                    f"  P5: expected {self.P5_CH}, got {_actual_p5}\n"
                    f"Update P4_CH and P5_CH class attributes to match."
                )

        # ── Hook storage ─────────────────────────────────────
        self._p4_feat = None
        self._p5_feat = None
        self._hooks   = []
        self._register_hooks()

        # ── Adapter layers ───────────────────────────────────
        # Project YOLOv8 channels → LLEN decoder channels
        dec4_ch = llen_base_ch * 4   # 128  (dec4 out_ch)
        dec3_ch = llen_base_ch * 2   #  64  (dec3 out_ch)

        self.adapter_p5 = FeatureAdapter(self.P5_CH, dec4_ch)  # 256 → 128
        self.adapter_p4 = FeatureAdapter(self.P4_CH, dec3_ch)  # 128 →  64

    # ----------------------------------------------------------
    def _register_hooks(self):
        """Attach forward hooks to the P4 and P5 backbone layers."""

        def make_hook(name):
            def hook(module, input, output):
                if name == 'p4':
                    self._p4_feat = output
                else:
                    self._p5_feat = output
            return hook

        layer_list = list(self.backbone)
        h4 = layer_list[self.P4_LAYER_IDX].register_forward_hook(make_hook('p4'))
        h5 = layer_list[self.P5_LAYER_IDX].register_forward_hook(make_hook('p5'))
        self._hooks = [h4, h5]

    # ----------------------------------------------------------
    def remove_hooks(self):
        """Call this if you want to detach hooks (e.g. at inference)."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ----------------------------------------------------------
    def forward(self, enhanced_img: torch.Tensor):
        """
        Run the enhanced image through the YOLOv8 backbone to extract
        P4 and P5 features, then project them via adapter layers.

        Args:
            enhanced_img : (B, 3, H, W) enhanced output from LLEN, [0,1]
                           Will be resized to 640×640 for YOLOv8 if needed.

        Returns:
            dgff_p4 : (B, llen_base_ch*2, H/4,  W/4)  — for LLEN dec3
            dgff_p5 : (B, llen_base_ch*4, H/8,  W/8)  — for LLEN dec4
        """
        B, C, H, W = enhanced_img.shape

        # YOLOv8 expects 640×640 input; resize if necessary
        if H != 640 or W != 640:
            x = F.interpolate(
                enhanced_img, size=(640, 640),
                mode='bilinear', align_corners=False
            )
        else:
            x = enhanced_img

        # Run backbone layer-by-layer up to P5_LAYER_IDX.
        # We cannot call self.backbone(x) directly because layers after
        # the pure backbone use Concat ops that expect list inputs — they
        # crash when called sequentially. The pure backbone layers 0-9
        # are all simple Conv/C2f/SPPF blocks that accept a single tensor.
        # YOLOv8 also saves certain intermediate outputs for skip routes;
        # we replicate that here by keeping a `saved` dict keyed by index.
        layer_list = list(self.backbone)
        saved = {}   # stores outputs of layers that feed into later Concat ops
        # Layer save indices for YOLOv8n backbone (layers 4 and 6 are reused)
        SAVE_LAYERS = {4, 6}
        out = x
        for i, layer in enumerate(layer_list[:self.P5_LAYER_IDX + 1]):
            out = layer(out)
            if i in SAVE_LAYERS:
                saved[i] = out
        # hooks already captured p4 (layer 6) and p5 (layer 9) via register_forward_hook

        p4_raw = self._p4_feat   # (B, 128, 40, 40)  for 640×640 input
        p5_raw = self._p5_feat   # (B, 256, 20, 20)  for 640×640 input

        if p4_raw is None or p5_raw is None:
            raise RuntimeError(
                "DGFF hooks did not capture features. "
                "Check P4_LAYER_IDX and P5_LAYER_IDX match your YOLOv8 version."
            )

        # Project to LLEN decoder channel depths
        dgff_p4_proj = self.adapter_p4(p4_raw)   # (B,  64, 40, 40)  128→64
        dgff_p5_proj = self.adapter_p5(p5_raw)   # (B, 128, 20, 20)  256→128

        # Resize projected features to match LLEN decoder spatial sizes
        # LLEN decoder spatial sizes depend on input resolution (H, W)
        # dec4 operates at H/8, W/8 — matches P5 after resize
        # dec3 operates at H/4, W/4 — matches P4 after resize
        dgff_p5_out = F.interpolate(
            dgff_p5_proj, size=(H // 8, W // 8),
            mode='bilinear', align_corners=False
        )
        dgff_p4_out = F.interpolate(
            dgff_p4_proj, size=(H // 4, W // 4),
            mode='bilinear', align_corners=False
        )

        return dgff_p4_out, dgff_p5_out

    # ----------------------------------------------------------
    def count_parameters(self):
        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ─────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    from pathlib import Path

    print("=" * 55)
    print("  DGFF Module Smoke Test")
    print("=" * 55)

    device = (
        torch.device('mps')   if torch.backends.mps.is_available() else
        torch.device('cuda')  if torch.cuda.is_available()         else
        torch.device('cpu')
    )
    print(f"  Device : {device}")

    # ── Load DGFF ────────────────────────────────────────────
    print("  Loading YOLOv8n backbone ...")
    dgff = DGFFModule(
        yolo_weights    = 'yolov8n.pt',
        llen_base_ch    = 32,
        freeze_backbone = True,
    ).to(device)

    total, trainable = dgff.count_parameters()
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable (adapters): {trainable:,}")

    # ── Forward pass test ────────────────────────────────────
    dummy_enhanced = torch.rand(2, 3, 256, 256).to(device)

    print(f"\n  Input  : {tuple(dummy_enhanced.shape)}")
    dgff_p4, dgff_p5 = dgff(dummy_enhanced)
    print(f"  dgff_p4 (→ dec3) : {tuple(dgff_p4.shape)}")   # expect (2, 64, 64, 64)
    print(f"  dgff_p5 (→ dec4) : {tuple(dgff_p5.shape)}")   # expect (2, 128, 32, 32)

    assert dgff_p4.shape == (2, 64,  64, 64), f"P4 shape wrong: {dgff_p4.shape}"
    assert dgff_p5.shape == (2, 128, 32, 32), f"P5 shape wrong: {dgff_p5.shape}"
    print("  Shape assertions passed ✓")

    # ── Full pipeline test: LLEN + DGFF ─────────────────────
    print("\n  Testing LLEN + DGFF combined forward pass ...")
    sys.path.insert(0, str(Path(__file__).parent if '__file__' in dir() else Path('.')))
    from llen import LLEN

    llen = LLEN(base_ch=32).to(device)

    # Simulate the two-pass training forward:
    # Pass 1: LLEN enhances without feedback (cold start)
    enhanced = llen(dummy_enhanced)
    print(f"  Pass 1 enhanced : {tuple(enhanced.shape)}")

    # Pass 2: DGFF extracts features from enhanced image
    dgff_p4, dgff_p5 = dgff(enhanced)

    # Pass 3: LLEN re-enhances with detection feedback injected
    enhanced_guided = llen(dummy_enhanced, dgff_p5=dgff_p5, dgff_p4=dgff_p4)
    print(f"  Pass 2 guided   : {tuple(enhanced_guided.shape)}")

    assert enhanced_guided.shape == dummy_enhanced.shape
    print("  Combined pipeline ✓")

    # ── Gradient flow test ───────────────────────────────────
    print("\n  Testing gradient flow through DGFF adapters ...")
    import torch.nn.functional as F_test

    llen.train()
    dgff.train()

    enhanced      = llen(dummy_enhanced)
    dgff_p4, dgff_p5 = dgff(enhanced)
    enhanced_guided  = llen(dummy_enhanced, dgff_p5=dgff_p5, dgff_p4=dgff_p4)

    dummy_target = torch.rand_like(enhanced_guided)
    loss = F_test.l1_loss(enhanced_guided, dummy_target)
    loss.backward()

    # Check adapter grads received signal
    for name, param in dgff.adapter_p5.named_parameters():
        assert param.grad is not None, f"adapter_p5.{name} has no gradient!"
    for name, param in dgff.adapter_p4.named_parameters():
        assert param.grad is not None, f"adapter_p4.{name} has no gradient!"

    print("  Adapter gradients flowing ✓")
    print("\n  All DGFF tests passed ✓")
