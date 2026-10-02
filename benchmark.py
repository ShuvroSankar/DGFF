import time
import torch
import torch.nn as nn
from ultralytics import YOLO
from thop import profile

from models import LLEN, DGFFModule

torch.manual_seed(0)

IMG_SIZE = 256   # matches training crop size in the paper
N_WARMUP = 5
N_TIMED = 30

device = torch.device('cpu')

print("=" * 70)
print("1. PARAMETER COUNTS")
print("=" * 70)

llen = LLEN(base_ch=32).to(device).eval()
dgff = DGFFModule(yolo_weights='yolov8n.pt', llen_base_ch=32, freeze_backbone=True).to(device).eval()

llen_params = sum(p.numel() for p in llen.parameters())
dgff_total, dgff_trainable = dgff.count_parameters()
backbone_params = sum(p.numel() for p in dgff.backbone.parameters())
adapter_params = dgff_total - backbone_params

print(f"LLEN total params            : {llen_params:,}")
print(f"DGFF backbone (frozen) params: {backbone_params:,}")
print(f"DGFF adapter (trainable) params: {adapter_params:,}")
print(f"DGFF module total params     : {dgff_total:,}")

# Full YOLOv8n detector (backbone+neck+head) for actual detection inference
yolo_full = YOLO('yolov8n.pt')
yolo_det_params = sum(p.numel() for p in yolo_full.model.parameters())
print(f"Full YOLOv8n detector params : {yolo_det_params:,}")

print()
print("=" * 70)
print("2. FLOPs (thop, single image, {}x{})".format(IMG_SIZE, IMG_SIZE))
print("=" * 70)

dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)

# LLEN alone (Pass-1-equivalent / inference-time enhancement)
macs_llen, params_llen = profile(llen, inputs=(dummy,), verbose=False)
print(f"LLEN forward            : {macs_llen/1e9:.3f} GMACs  ({2*macs_llen/1e9:.3f} GFLOPs)")

# DGFF module alone (backbone truncated to 10 layers + 3 adapters) -- TRAINING ONLY
macs_dgff, params_dgff = profile(dgff, inputs=(dummy,), verbose=False)
print(f"DGFF module forward (train-only): {macs_dgff/1e9:.3f} GMACs  ({2*macs_dgff/1e9:.3f} GFLOPs)")

# Full YOLOv8n detector alone (inference-time detection step)
yolo_model = yolo_full.model.to(device).eval()
macs_yolo, params_yolo = profile(yolo_model, inputs=(dummy,), verbose=False)
print(f"Full YOLOv8n detector forward   : {macs_yolo/1e9:.3f} GMACs  ({2*macs_yolo/1e9:.3f} GFLOPs)")

print()
print("Pipeline totals:")
raw_flops = 2 * macs_yolo
llen_only_flops = 2 * macs_llen + 2 * macs_yolo
dgff_train_flops = 2 * macs_llen + 2 * macs_dgff + 2 * macs_llen + 2 * macs_yolo  # pass1 LLEN + DGFF + pass2 LLEN + detect
dgff_infer_flops = llen_only_flops  # DGFF discarded at inference -> identical to LLEN-only

print(f"  Raw -> Detect                        : {raw_flops/1e9:.3f} GFLOPs")
print(f"  LLEN-only -> Detect (inference)       : {llen_only_flops/1e9:.3f} GFLOPs")
print(f"  DGFF TRAINING (2-pass + adapters)     : {dgff_train_flops/1e9:.3f} GFLOPs")
print(f"  DGFF INFERENCE (adapters discarded)   : {dgff_infer_flops/1e9:.3f} GFLOPs  (identical arch. to LLEN-only)")

print()
print("=" * 70)
print("3. CPU LATENCY  (single-core sandbox CPU -- relative comparison only)")
print("=" * 70)


def time_forward(fn, n_warmup=N_WARMUP, n_timed=N_TIMED):
    with torch.no_grad():
        for _ in range(n_warmup):
            fn()
        times = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    times.sort()
    return sum(times) / len(times), times[len(times) // 2]


mean_llen, med_llen = time_forward(lambda: llen(dummy))
mean_yolo, med_yolo = time_forward(lambda: yolo_model(dummy))
mean_dgff, med_dgff = time_forward(lambda: dgff(dummy))

print(f"LLEN forward             : mean {mean_llen*1000:.2f} ms | median {med_llen*1000:.2f} ms")
print(f"Full YOLOv8n forward      : mean {mean_yolo*1000:.2f} ms | median {med_yolo*1000:.2f} ms")
print(f"DGFF module forward (train-only) : mean {mean_dgff*1000:.2f} ms | median {med_dgff*1000:.2f} ms")

raw_latency = mean_yolo
llen_only_latency = mean_llen + mean_yolo
dgff_infer_latency = llen_only_latency  # identical arch at inference
dgff_train_latency = 2 * mean_llen + mean_dgff + mean_yolo  # 2-pass LLEN + DGFF extraction/adapters + detect

print()
print("Pipeline latency (mean, ms/image):")
print(f"  Raw -> Detect                     : {raw_latency*1000:.2f} ms  ({1/raw_latency:.1f} FPS)")
print(f"  LLEN-only -> Detect (inference)    : {llen_only_latency*1000:.2f} ms  ({1/llen_only_latency:.1f} FPS)")
print(f"  DGFF INFERENCE (adapters discarded): {dgff_infer_latency*1000:.2f} ms  ({1/dgff_infer_latency:.1f} FPS)")
print(f"  DGFF TRAINING (2-pass, for ref.)   : {dgff_train_latency*1000:.2f} ms  ({1/dgff_train_latency:.1f} FPS)")

print()
print("=" * 70)
print("4. SANITY CHECK vs paper-reported parameter counts")
print("=" * 70)
print(f"Paper claims LLEN = 4,844,803       | measured = {llen_params:,}  | match={llen_params==4844803}")
print(f"Paper claims DGFF total = 1,359,120 | measured = {dgff_total:,}  | match={dgff_total==1359120}")
print(f"Paper claims adapters   = 86,464    | measured = {adapter_params:,}  | match={adapter_params==86464}")
print(f"Paper claims backbone   = 1,272,656 | measured = {backbone_params:,}  | match={backbone_params==1272656}")
