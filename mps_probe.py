"""
mps_probe.py — answers ONE question: will the Table 6 ablation
(real_ablation_cell.py, which trains LLEN+DGFFModule directly, NOT via
Ultralytics) run on MPS or fall back to CPU on this machine/environment?

This does NOT test Ultralytics detector training (that's a separate code
path and separate question, already known to be slow on this setup).
This tests the actual custom training loop the ablation uses.

Strictly time-bounded: 10 iterations only, so this finishes in well under
a minute even in the worst case. Needs models.py in the same directory.
"""

import time
import torch

print(f"torch version: {torch.__version__}")
print(f"MPS built:     {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

if not torch.backends.mps.is_available():
    print("\nMPS is NOT available in this environment. The ablation will run "
          "on CPU. Given today's CPU timings for a much smaller model were "
          "already very slow, expect the ablation to be impractically slow "
          "too -- don't start it blind. Consider a different conda env/torch "
          "build (this one is on Python 3.14.2 / torch 2.10.0, both very new "
          "-- an older, more established torch build may have better MPS "
          "support) before committing to a long run.")
    raise SystemExit

device = torch.device('mps')
print(f"\nUsing device: {device}")

from models import LLEN, DGFFModule  # same file used for benchmark.py earlier

BATCH = 4          # matches paper's Phase 1 batch size
CROP = 256          # matches paper's Phase 1 crop size
N_ITERS = 10

print(f"\nBuilding LLEN + DGFFModule on {device}...")
llen = LLEN(base_ch=32).to(device)
dgff = DGFFModule(yolo_weights='yolov8n.pt', llen_base_ch=32, freeze_backbone=True).to(device)

trainable = list(llen.parameters()) + list(dgff.adapter_p3.parameters()) + \
            list(dgff.adapter_p4.parameters()) + list(dgff.adapter_p5.parameters())
optimiser = torch.optim.Adam(trainable, lr=1e-4)
l1 = torch.nn.L1Loss()

dummy_low = torch.rand(BATCH, 3, CROP, CROP, device=device)
dummy_high = torch.rand(BATCH, 3, CROP, CROP, device=device)

print(f"Running {N_ITERS} timed training iterations "
      f"(batch={BATCH}, crop={CROP}x{CROP}, matches paper's Phase 1 settings)...")

# warmup (first MPS call compiles kernels -- don't count this)
enhanced = llen(dummy_low)
p4, p5, p3 = dgff(enhanced)
guided = llen(dummy_low, dgff_p5=p5, dgff_p4=p4, dgff_p3=p3)
loss = l1(guided, dummy_high)
loss.backward()
torch.mps.synchronize()

times = []
for i in range(N_ITERS):
    torch.mps.synchronize()
    t0 = time.perf_counter()

    optimiser.zero_grad()
    enhanced = llen(dummy_low)
    p4, p5, p3 = dgff(enhanced)
    guided = llen(dummy_low, dgff_p5=p5, dgff_p4=p4, dgff_p3=p3)
    loss = l1(guided, dummy_high)
    loss.backward()
    optimiser.step()

    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  iter {i+1}/{N_ITERS}: {dt:.3f}s")

mean_iter = sum(times) / len(times)
print(f"\nMean time/iteration: {mean_iter:.3f}s")

# ── extrapolate to full ablation cost ──
TRAIN_PAIRS = 485          # paper's LOL train set size
iters_per_epoch = TRAIN_PAIRS / BATCH
ablation_epochs = 50
n_configs = 4
total_iters = iters_per_epoch * ablation_epochs * n_configs
est_seconds = total_iters * mean_iter

print(f"\n=== Extrapolated full ablation cost (4 configs x {ablation_epochs} epochs) ===")
print(f"  ~{iters_per_epoch:.0f} iterations/epoch x {ablation_epochs} epochs x {n_configs} configs "
      f"= {total_iters:.0f} total iterations")
print(f"  Estimated total time: {est_seconds/3600:.1f} hours")
print(f"  (paper's Phase 1: 200 epochs, single config, reported as 7.6h on MPS -- "
      f"this is a rough sanity cross-check on the estimate above)")
