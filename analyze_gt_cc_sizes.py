"""
Scan dataset GT masks for a given split and report connected-component
pixel-size distribution at the eval resolution (img_size x img_size,
NEAREST resize — same as transforms.Resize in test.py).

Usage:
    python analyze_gt_cc_sizes.py \
        --data_root ../dataset_2classes --split test --img_size 512 \
        --out gt_cc_sizes_test_512.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='../dataset_2classes')
    ap.add_argument('--split', default='test')
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--out', default='gt_cc_sizes.json')
    args = ap.parse_args()

    masks_dir = Path(args.data_root) / 'masks' / args.split
    paths = sorted(p for p in masks_dir.iterdir() if p.suffix == '.png')
    print(f'Scanning {len(paths)} masks in {masks_dir} @ {args.img_size}x{args.img_size}')

    per_slice = []
    all_sizes: list[int] = []
    slices_with_nodule = 0
    total_components = 0

    for p in paths:
        gt = Image.open(p)
        if gt.size != (args.img_size, args.img_size):
            gt = gt.resize((args.img_size, args.img_size), Image.NEAREST)
        arr = (np.array(gt) > 0).astype(np.uint8)
        if arr.sum() == 0:
            per_slice.append({'mask': p.name, 'sizes': []})
            continue
        labels, n = cc_label(arr)
        if n == 0:
            per_slice.append({'mask': p.name, 'sizes': []})
            continue
        sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:].tolist()
        per_slice.append({'mask': p.name, 'sizes': sizes})
        all_sizes.extend(sizes)
        slices_with_nodule += 1
        total_components += n

    a = np.array(all_sizes, dtype=np.int64) if all_sizes else np.array([0])
    summary = {
        'split': args.split,
        'img_size': args.img_size,
        'n_slices': len(paths),
        'n_slices_with_nodule': slices_with_nodule,
        'n_components': total_components,
        'min': int(a.min()) if all_sizes else 0,
        'max': int(a.max()) if all_sizes else 0,
        'mean': float(a.mean()) if all_sizes else 0.0,
        'median': float(np.median(a)) if all_sizes else 0.0,
        'percentiles': {
            f'p{q}': float(np.percentile(a, q))
            for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        } if all_sizes else {},
        'small_counts': {
            f'<= {thr} px': int((a <= thr).sum())
            for thr in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
        } if all_sizes else {},
    }

    out = {'summary': summary, 'per_slice': per_slice}
    Path(args.out).write_text(json.dumps(out))
    print(json.dumps(summary, indent=2))
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
