from ultralytics import YOLO

# Candidates found on disk, with a guess at what each one is based on path naming.
# Update the yaml paths if yours differ.
candidates = [
    ('checkpoints/best.pt',
     None,  # not a detector -- expect this to fail to load as YOLO, or be meaningless
     'Guess: LLEN enhancement network (Phase 1), not a detector'),

    ('runs/detect/runs/detect/domain_detectors/llen_detector/weights/best.pt',
     'detectors/llen/data.yaml',
     'Guess: LLEN-only domain-matched detector (paper reports mAP@0.5 = 0.469)'),

    ('runs/detect/runs/detect/domain_detectors/dgff_detector/weights/best.pt',
     'detectors/dgff/data.yaml',
     'Guess: DGFF domain-matched detector (paper reports mAP@0.5 = 0.474)'),
]

PAPER_NUMBERS = {
    'detectors/llen/data.yaml': 0.469,
    'detectors/dgff/data.yaml': 0.474,
}

print(f"{'Checkpoint':<75} {'mAP@0.5':>10} {'Paper':>8} {'Diff':>8}")
print("-" * 105)

for path, yaml, note in candidates:
    print(f"\n{note}")
    if yaml is None:
        try:
            m = YOLO(path)
            print(f"  Loaded as YOLO OK -- unexpected if this is really the "
                  f"enhancement network. Check what it actually is.")
        except Exception as e:
            print(f"  Failed to load as a YOLO detector ({type(e).__name__}: {e}) "
                  f"-- confirms this is NOT a detector checkpoint. Expected.")
        continue

    try:
        m = YOLO(path)
        r = m.val(data=yaml, split='val', verbose=False)
        map50 = r.box.map50
        paper = PAPER_NUMBERS.get(yaml)
        diff = abs(map50 - paper) if paper is not None else None
        diff_str = f"{diff:.4f}" if diff is not None else "n/a"
        print(f"  {path:<73} {map50:>10.4f} {paper:>8} {diff_str:>8}")
        if diff is not None and diff < 0.01:
            print(f"  -> MATCH: this is very likely the checkpoint used in the paper.")
        elif diff is not None:
            print(f"  -> Does not match closely. Either wrong file, wrong yaml/split, "
                  f"or this checkpoint predates your final reported run.")
    except Exception as e:
        print(f"  Failed: {type(e).__name__}: {e}")
