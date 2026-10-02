import os
import glob
import yaml
import numpy as np
from ultralytics import YOLO

with open('detectors/llen/data.yaml') as f:
    cfg = yaml.safe_load(f)

root = cfg['path']
img_dir = os.path.join(root, cfg['val'])
lbl_dir = img_dir.replace('images', 'labels')

print(f"root:    {root}")
print(f"img_dir: {img_dir}")
print(f"lbl_dir: {lbl_dir}")
print(f"img_dir exists: {os.path.isdir(img_dir)}")
print(f"lbl_dir exists: {os.path.isdir(lbl_dir)}")

img_paths = sorted(glob.glob(os.path.join(img_dir, '*')))
lbl_paths = sorted(glob.glob(os.path.join(lbl_dir, '*')))
print(f"\nfiles found in img_dir: {len(img_paths)}")
print(f"files found in lbl_dir: {len(lbl_paths)}")
print(f"first 3 img files: {[os.path.basename(p) for p in img_paths[:3]]}")
print(f"first 3 lbl files: {[os.path.basename(p) for p in lbl_paths[:3]]}")

if not img_paths:
    print("\nNo images found -- stop here, the glob pattern or path is wrong.")
    raise SystemExit

# ── inspect ONE image in detail ──
test_img = img_paths[0]
fname = os.path.splitext(os.path.basename(test_img))[0]
lbl_path = os.path.join(lbl_dir, fname + '.txt')

print(f"\n--- Inspecting: {fname} ---")
print(f"expected label path: {lbl_path}")
print(f"label file exists: {os.path.exists(lbl_path)}")

if os.path.exists(lbl_path):
    with open(lbl_path) as f:
        content = f.read()
    print(f"raw label file content:\n{content[:500]}")
else:
    print("Label file does NOT exist at that path -- this is likely the bug. "
          "Check what's actually inside lbl_dir below:")
    print(os.listdir(lbl_dir)[:10] if os.path.isdir(lbl_dir) else "lbl_dir doesn't exist")

# ── run the model on just this one image, print raw prediction format ──
print(f"\n--- Model prediction on same image ---")
model = YOLO('runs/detect/runs/detect/domain_detectors/llen_detector/weights/best.pt')
r = list(model.predict(source=test_img, conf=0.001, iou=0.7, verbose=False))[0]
print(f"r.path: {r.path}")
print(f"r.orig_shape: {r.orig_shape}")
if r.boxes is not None and len(r.boxes) > 0:
    print(f"num predictions: {len(r.boxes)}")
    print(f"first 3 xyxyn (normalized): \n{r.boxes.xyxyn[:3].cpu().numpy()}")
    print(f"first 3 xyxy (pixel): \n{r.boxes.xyxy[:3].cpu().numpy()}")
    print(f"first 3 conf: {r.boxes.conf[:3].cpu().numpy()}")
    print(f"first 3 cls: {r.boxes.cls[:3].cpu().numpy()}")
else:
    print("No predictions at all for this image (unexpected at conf=0.001).")
