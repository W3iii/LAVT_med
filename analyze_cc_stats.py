"""
Analyze cc_stats.json produced by test.py --cc_stats_json:
report pred CC size distribution split into TP/FP at Dice 0.3 and 0.5,
and simulate size-based FP filtering at a range of cutoffs.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(sizes: np.ndarray, name: str) -> None:
    if sizes.size == 0:
        print(f'  {name}: (empty)')
        return
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pcts = np.percentile(sizes, qs)
    print(f'  {name}: n={sizes.size:>5d}  min={sizes.min():>4d}  '
          f'max={sizes.max():>5d}  mean={sizes.mean():>7.1f}  median={np.median(sizes):>6.1f}')
    print('    pcts ' + '  '.join(f'p{q}={int(p):d}' for q, p in zip(qs, pcts)))


def cumulative_le(sizes: np.ndarray, cutoffs: list[int]) -> list[int]:
    return [int((sizes <= c).sum()) for c in cutoffs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cc_stats', default='./pred_results/lavt_one_ln_v1/cc_stats.json')
    ap.add_argument('--thr', default='0.3', help='match threshold key to split TP/FP')
    args = ap.parse_args()

    data = json.loads(Path(args.cc_stats).read_text())
    metric = data.get('match_metric', 'iou')
    print(f'img_size = {data["img_size"]}, match_metric = {metric}, '
          f'thresholds in file = {data["thresholds"]}')
    slices = data['slices']
    print(f'total slices: {len(slices)}')

    n_normal = sum(1 for s in slices if s['is_normal'])
    n_nodule = len(slices) - n_normal
    print(f'  normal: {n_normal}, nodule: {n_nodule}')

    for thr_key in ('0.3', '0.5'):
        # Collect pred CC sizes split by TP/FP at this threshold.
        pred_tp, pred_fp_nodule, pred_fp_normal = [], [], []
        gt_matched, gt_missed = [], []
        for s in slices:
            ps = s['pred_sizes']
            pm = s['pred_matched'][thr_key]
            for size, hit in zip(ps, pm):
                if hit:
                    pred_tp.append(size)
                elif s['is_normal']:
                    pred_fp_normal.append(size)
                else:
                    pred_fp_nodule.append(size)
            gs = s['gt_sizes']
            gm = s['gt_matched'][thr_key]
            for size, hit in zip(gs, gm):
                (gt_matched if hit else gt_missed).append(size)

        tp = np.array(pred_tp, dtype=np.int64)
        fp_nod = np.array(pred_fp_nodule, dtype=np.int64)
        fp_norm = np.array(pred_fp_normal, dtype=np.int64)
        fp_all = np.concatenate([fp_nod, fp_norm]) if (fp_nod.size or fp_norm.size) else np.array([], dtype=np.int64)
        gt_m = np.array(gt_matched, dtype=np.int64)
        gt_x = np.array(gt_missed, dtype=np.int64)

        print(f'\n=== {metric.upper()} threshold {thr_key} ===')
        print('Pred CC size distributions:')
        summarize(tp, 'TP        ')
        summarize(fp_nod, 'FP@nodule ')
        summarize(fp_norm, 'FP@normal ')
        summarize(fp_all, 'FP all    ')
        print('GT CC size distributions:')
        summarize(gt_m, 'GT matched')
        summarize(gt_x, 'GT missed ')

        cutoffs = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
        tp_cum = cumulative_le(tp, cutoffs)
        fp_cum = cumulative_le(fp_all, cutoffs)
        fp_nod_cum = cumulative_le(fp_nod, cutoffs)
        fp_norm_cum = cumulative_le(fp_norm, cutoffs)
        gt_cum = cumulative_le(gt_m, cutoffs)

        print('\nFilter simulation (drop pred CC with size <= cutoff):')
        print(f'  {"cut":>4} | {"TP_lost":>7} ({ "%":>5}) | {"FP_kill":>7} ({ "%":>5}) | '
              f'{"FP_kill_nod":>11} {"FP_kill_norm":>12}')
        for c, tl, fl, fln, flnorm in zip(cutoffs, tp_cum, fp_cum, fp_nod_cum, fp_norm_cum):
            tp_pct = 100.0 * tl / tp.size if tp.size else 0.0
            fp_pct = 100.0 * fl / fp_all.size if fp_all.size else 0.0
            print(f'  {c:>4d} | {tl:>7d} ({tp_pct:>4.1f}%) | {fl:>7d} ({fp_pct:>4.1f}%) | '
                  f'{fln:>11d} {flnorm:>12d}')

        # Effect on per-nodule recall/precision at this match threshold.
        # Recall uses GT side; size filter does not remove GT, only pred — so
        # missed GT that was already matched would become unmatched only if the
        # pred CC matching it was filtered. Approximate: a TP becomes FN when
        # its pred CC is filtered (1:1 mapping at matching time).
        n_gt_pos = gt_m.size + gt_x.size
        n_tp = tp.size
        n_fp = fp_all.size
        print('\nProjected per-nodule metrics after filter:')
        print(f'  cut | TP_keep FP_keep | Recall  Precision')
        for c in cutoffs:
            tp_keep = int((tp > c).sum())
            fp_keep = int((fp_all > c).sum())
            recall = 100.0 * tp_keep / n_gt_pos if n_gt_pos else 0.0
            precision = 100.0 * tp_keep / (tp_keep + fp_keep) if (tp_keep + fp_keep) else 0.0
            print(f'  {c:>3d} | {tp_keep:>7d} {fp_keep:>7d} | {recall:>5.2f}%  {precision:>6.2f}%')

        # Baseline recall/precision (no filter).
        baseline_recall = 100.0 * n_tp / n_gt_pos if n_gt_pos else 0.0
        baseline_precision = 100.0 * n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
        print(f'  baseline (no filter): Recall={baseline_recall:.2f}%  '
              f'Precision={baseline_precision:.2f}%  '
              f'(TP={n_tp}, FP={n_fp}, GT_total={n_gt_pos})')


if __name__ == '__main__':
    main()
