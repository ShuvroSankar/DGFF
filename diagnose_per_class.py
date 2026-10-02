import numpy as np

CLASS_NAMES = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair',
               'Cup', 'Dog', 'Motorbike', 'People', 'Table']

cache = np.load('bootstrap_stats_cache.npz', allow_pickle=True)
llen_stats = cache['llen_stats'].item()
dgff_stats = cache['dgff_stats'].item()


def per_class_recall(stats, tag):
    tp_counts = {}
    gt_counts = {}
    for fname, d in stats.items():
        for c, n in d['gt_counts'].items():
            gt_counts[c] = gt_counts.get(c, 0) + n
        for c, tp in zip(d['cls'], d['tp']):
            if tp:
                tp_counts[c] = tp_counts.get(c, 0) + 1

    print(f"\n=== {tag}: per-class recall at IoU>=0.5 (diagnostic only, not AP) ===")
    print(f"{'Class':<12}{'GT':>8}{'TP':>8}{'Recall':>10}")
    total_gt, total_tp = 0, 0
    for c in sorted(gt_counts):
        gt = gt_counts[c]
        tp = tp_counts.get(c, 0)
        name = CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f"cls_{c}"
        print(f"{name:<12}{gt:>8}{tp:>8}{tp/gt:>10.3f}")
        total_gt += gt
        total_tp += tp
    print(f"{'TOTAL':<12}{total_gt:>8}{total_tp:>8}{total_tp/total_gt:>10.3f}")
    return gt_counts, tp_counts


llen_gt, llen_tp = per_class_recall(llen_stats, 'LLEN')
dgff_gt, dgff_tp = per_class_recall(dgff_stats, 'DGFF')

print("\n=== Per-class recall gap between LLEN and DGFF ===")
print(f"{'Class':<12}{'LLEN recall':>14}{'DGFF recall':>14}{'Diff':>10}")
for c in sorted(llen_gt):
    name = CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f"cls_{c}"
    r_llen = llen_tp.get(c, 0) / llen_gt[c]
    r_dgff = dgff_tp.get(c, 0) / dgff_gt.get(c, 1)
    print(f"{name:<12}{r_llen:>14.3f}{r_dgff:>14.3f}{r_dgff - r_llen:>10.3f}")
