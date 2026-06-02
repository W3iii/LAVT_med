import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from scipy.ndimage import label as cc_label

from lib import segmentation
from data.dataset_lung_nodule import LungNoduleDataset
import transforms as T
import utils

NODULE_DICE_THRS = (0.3, 0.5)


def get_input_size(args):
    if args.img_size is not None:
        return args.img_size, args.img_size
    return args.img_h, args.img_w


def format_input_size(args):
    img_h, img_w = get_input_size(args)
    return f'{img_w}x{img_h} (W x H)'


def get_transform(args):
    img_h, img_w = get_input_size(args)
    return T.Compose([
        T.Resize(img_h, img_w),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    """1-pixel boundary of a binary mask via 4-connected erosion."""
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m)
    padded = np.pad(m, 1, mode='constant', constant_values=False)
    eroded = (padded[:-2, 1:-1] & padded[2:, 1:-1] &
              padded[1:-1, :-2] & padded[1:-1, 2:])
    return m & ~eroded


def _per_object_dice_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray,
                            n_gt: int, n_pred: int) -> np.ndarray:
    """Dice between every GT component and every pred component."""
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float64)
    flat = gt_labels.ravel().astype(np.int64) * (n_pred + 1) \
        + pred_labels.ravel().astype(np.int64)
    bins = np.bincount(flat, minlength=(n_gt + 1) * (n_pred + 1))
    inter = bins.reshape(n_gt + 1, n_pred + 1)[1:, 1:].astype(np.float64)
    gt_sizes = np.bincount(gt_labels.ravel(), minlength=n_gt + 1)[1:].astype(np.float64)
    pred_sizes = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)[1:].astype(np.float64)
    denom = gt_sizes[:, None] + pred_sizes[None, :]
    return np.where(denom > 0, 2.0 * inter / denom, 0.0)


def _greedy_match(score_mat: np.ndarray, thr: float) -> tuple:
    """Match GT/pred pairs greedily by descending score at threshold thr.

    The score matrix can be IoU, Dice, or any overlap measure. Returns
    (tp, fn, fp, matched_gt, matched_pred); the boolean arrays mark which
    GT/pred components got paired.
    """
    n_gt, n_pred = score_mat.shape
    matched_gt = np.zeros(n_gt, dtype=bool)
    matched_pred = np.zeros(n_pred, dtype=bool)
    if n_gt == 0:
        return 0, 0, n_pred, matched_gt, matched_pred
    if n_pred == 0:
        return 0, n_gt, 0, matched_gt, matched_pred
    pairs_i, pairs_j = np.where(score_mat >= thr)
    if pairs_i.size == 0:
        return 0, n_gt, n_pred, matched_gt, matched_pred
    order = np.argsort(-score_mat[pairs_i, pairs_j])
    pairs_i, pairs_j = pairs_i[order], pairs_j[order]
    for i, j in zip(pairs_i, pairs_j):
        if not matched_gt[i] and not matched_pred[j]:
            matched_gt[i] = True
            matched_pred[j] = True
    tp = int(matched_gt.sum())
    return tp, n_gt - tp, n_pred - int(matched_pred.sum()), matched_gt, matched_pred


def _save_overlay(image_path: Path, mask_path, pred: torch.Tensor,
                  out_path: Path, output_size: tuple) -> None:
    """
    Draw GT (green) and pred (red) contours on the same resized canvas
    used by validation/test transforms.
    """
    out_h, out_w = output_size
    img = Image.open(image_path).convert("L")
    if mask_path is None:
        gt = Image.new("L", img.size, 0)
    else:
        gt = Image.open(mask_path).convert("L")

    fit = T.Resize(out_h, out_w)
    img, gt = fit(img, gt)

    img_np = np.array(img, dtype=np.uint8)
    gt_np = np.array(gt, dtype=np.uint8)
    pred_np = pred.cpu().numpy().astype(np.uint8)
    if pred_np.shape != (out_h, out_w):
        pred_np = np.array(
            Image.fromarray(pred_np).resize((out_w, out_h), Image.NEAREST),
            dtype=np.uint8,
        )

    rgb = np.stack([img_np, img_np, img_np], axis=-1)
    rgb[_mask_outline(gt_np)] = (0, 255, 0)
    rgb[_mask_outline(pred_np)] = (255, 0, 0)
    Image.fromarray(rgb).save(out_path)


@torch.no_grad()
def evaluate(model, data_loader, device, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    output_size = get_input_size(args)

    if args.save_pred:
        pred_root = Path(args.pred_dir)
        (pred_root / "nodule").mkdir(parents=True, exist_ok=True)
        (pred_root / "normal").mkdir(parents=True, exist_ok=True)
        images_dir = Path(args.data_root) / "images" / args.split
        masks_dir = Path(args.data_root) / "masks" / args.split
        print(f'Saving overlays to {pred_root}/{{nodule,normal}}')

    dump_cc = bool(args.cc_stats_json)
    cc_stats_slices: list = [] if dump_cc else []
    want_meta = args.save_pred or dump_cc

    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    seg_correct = np.zeros(len(iou_thresholds), dtype=np.int64)

    iou_chunks = []
    cum_inter = torch.zeros((), dtype=torch.float64, device=device)
    cum_union = torch.zeros((), dtype=torch.float64, device=device)
    n_pos, n_neg, n_tn = 0, 0, 0

    # Per-nodule (connected-component) tallies at each Dice threshold,
    # tracked separately for nodule-only slices vs all slices.
    cc_all = {thr: [0, 0, 0] for thr in NODULE_DICE_THRS}    # tp, fn, fp
    cc_nod = {thr: [0, 0, 0] for thr in NODULE_DICE_THRS}

    # Per-nodule mean Dice (best-match, threshold-free)
    nodule_dice_sum = 0.0
    nodule_dice_count = 0

    # Clinical size bins at test resolution (~0.91mm/px effective):
    # <4mm≈15px, 6mm≈34px, 8mm≈61px (circle area = π*(d/2/spacing)²)
    SIZE_BINS   = [(1, 15), (15, 34), (34, 61), (61, int(1e9))]
    SIZE_LABELS = ['<4mm\n(Benign)', '4-6mm\n(Prob.Benign)',
                   '6-8mm\n(Prob.Suspicious)', '>8mm\n(Suspicious)']
    CAT_COLORS  = ['#999999', '#5a9e6f', '#e07b39', '#c0392b']
    # per threshold → per bin → [matched(TP), total(TP+FN)]
    size_recall = {thr: [[0, 0] for _ in SIZE_BINS] for thr in NODULE_DICE_THRS}
    # per threshold → per bin → FP count (binned by pred CC size)
    size_fp     = {thr: [0 for _ in SIZE_BINS] for thr in NODULE_DICE_THRS}

    # Filtered CC metrics: GT nodules >= GT_SIZE_MIN_PX only
    GT_SIZE_MIN_PX = 15
    cc_all_f = {thr: [0, 0, 0] for thr in NODULE_DICE_THRS}
    cc_nod_f = {thr: [0, 0, 0] for thr in NODULE_DICE_THRS}

    # FROC accumulators
    froc_cands: list = []   # (score, is_tp)
    froc_gt_total = 0
    froc_img_total = 0

    for batch in metric_logger.log_every(data_loader, 100, header):
        if want_meta:
            image, target, ann_batch = batch
        else:
            image, target = batch
            ann_batch = None

        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(image)
        prob = logits.softmax(dim=1)[:, 1]             # (B, H, W) float [0, 1]
        pred = logits.argmax(dim=1)                    # (B, H, W) {0, 1}

        if args.save_pred:
            for i in range(pred.shape[0]):
                img_name = ann_batch["image"][i]
                mask_name = ann_batch["mask"][i]
                is_normal = (mask_name == "empty")
                subdir = "normal" if is_normal else "nodule"
                stem = os.path.splitext(img_name)[0]
                out_path = pred_root / subdir / f"{stem}_overlay.png"
                gt_path = None if is_normal else (masks_dir / mask_name)
                _save_overlay(images_dir / img_name, gt_path,
                              pred[i], out_path, output_size)

        target_flat = target.flatten(1)
        pred_flat = pred.flatten(1)
        is_pos = target_flat.any(dim=1)

        inter = (pred_flat * target_flat).sum(dim=1).double()
        union = (pred_flat.sum(dim=1) + target_flat.sum(dim=1)).double() - inter
        iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))

        if is_pos.any():
            iou_pos = iou[is_pos]
            iou_chunks.append(iou_pos.cpu())
            cum_inter += inter[is_pos].sum()
            cum_union += union[is_pos].sum()
            n_pos += int(is_pos.sum().item())
            iou_np = iou_pos.cpu().numpy()
            for k, thr in enumerate(iou_thresholds):
                seg_correct[k] += int((iou_np >= thr).sum())

        neg = ~is_pos
        if neg.any():
            pred_neg_sum = pred_flat[neg].sum(dim=1)
            n_tn += int((pred_neg_sum == 0).sum().item())
            n_neg += int(neg.sum().item())

        # Per-nodule (connected-component) match — done on CPU per sample.
        pred_cpu = pred.cpu().numpy().astype(np.uint8)
        prob_cpu = prob.cpu().numpy()                  # float32 (B, H, W)
        target_cpu = target.cpu().numpy().astype(np.uint8)
        for b in range(pred_cpu.shape[0]):
            gt_labels, n_gt = cc_label(target_cpu[b])
            pr_labels, n_pred = cc_label(pred_cpu[b])
            dice_mat = _per_object_dice_matrix(gt_labels, pr_labels, n_gt, n_pred)

            # Compute GT sizes once, reused below
            gt_sizes_arr = (np.bincount(gt_labels.ravel(), minlength=n_gt + 1)[1:].astype(int)
                            if n_gt > 0 else np.array([], dtype=int))

            # Per-nodule best-match Dice (GT nodules only)
            if n_gt > 0:
                best_per_gt = dice_mat.max(axis=1) if n_pred > 0 else np.zeros(n_gt)
                nodule_dice_sum += float(best_per_gt.sum())
                nodule_dice_count += n_gt

            per_thr_match = {}
            for thr in NODULE_DICE_THRS:
                tp, fn, fp, mg, mp = _greedy_match(dice_mat, thr)
                cc_all[thr][0] += tp; cc_all[thr][1] += fn; cc_all[thr][2] += fp
                if n_gt > 0:
                    cc_nod[thr][0] += tp; cc_nod[thr][1] += fn; cc_nod[thr][2] += fp
                per_thr_match[thr] = (mg, mp)

            # Filtered CC match: exclude GT nodules < GT_SIZE_MIN_PX
            valid_gt = gt_sizes_arr >= GT_SIZE_MIN_PX  # bool (n_gt,)
            n_gt_f = int(valid_gt.sum())
            if n_gt > 0 and n_gt_f > 0:
                gt_labels_f = np.zeros_like(gt_labels)
                new_idx = 0
                for i in range(n_gt):
                    if valid_gt[i]:
                        new_idx += 1
                        gt_labels_f[gt_labels == i + 1] = new_idx
                dice_mat_f = _per_object_dice_matrix(gt_labels_f, pr_labels, n_gt_f, n_pred)
            else:
                dice_mat_f = np.zeros((n_gt_f, n_pred))
            for thr in NODULE_DICE_THRS:
                tp, fn, fp, mg, mp = _greedy_match(dice_mat_f, thr)
                cc_all_f[thr][0] += tp; cc_all_f[thr][1] += fn; cc_all_f[thr][2] += fp
                if n_gt_f > 0:
                    cc_nod_f[thr][0] += tp; cc_nod_f[thr][1] += fn; cc_nod_f[thr][2] += fp

            # Size-stratified recall (GT nodule area → bin)
            if n_gt > 0:
                for thr in NODULE_DICE_THRS:
                    mg = per_thr_match[thr][0]
                    for i, sz in enumerate(gt_sizes_arr):
                        for bi, (lo, hi) in enumerate(SIZE_BINS):
                            if lo <= sz < hi:
                                size_recall[thr][bi][1] += 1
                                if mg[i]:
                                    size_recall[thr][bi][0] += 1
                                break

            # Size-stratified FP (pred CC area → bin, for per-bin precision)
            if n_pred > 0:
                pred_sizes_arr = np.bincount(pr_labels.ravel(),
                                             minlength=n_pred + 1)[1:].astype(int)
                for thr in NODULE_DICE_THRS:
                    mp = per_thr_match[thr][1]
                    for j in range(n_pred):
                        if not mp[j]:
                            sz = pred_sizes_arr[j]
                            for bi, (lo, hi) in enumerate(SIZE_BINS):
                                if lo <= sz < hi:
                                    size_fp[thr][bi] += 1
                                    break

            # FROC: extract candidates at low threshold, score by max prob
            if args.froc_json:
                froc_img_total += 1
                froc_gt_total += n_gt
                cand_mask = (prob_cpu[b] > args.froc_extract_thr).astype(np.uint8)
                pr_froc_labels, n_froc_pred = cc_label(cand_mask)
                if n_froc_pred > 0:
                    scores = np.array([prob_cpu[b][pr_froc_labels == c].max()
                                       for c in range(1, n_froc_pred + 1)])
                    froc_dice = _per_object_dice_matrix(
                        gt_labels, pr_froc_labels, n_gt, n_froc_pred)
                    is_tp = np.zeros(n_froc_pred, dtype=bool)
                    gt_matched = np.zeros(max(n_gt, 1), dtype=bool)
                    for j in np.argsort(-scores):
                        if n_gt > 0:
                            best_gt = int(froc_dice[:, j].argmax())
                            if (froc_dice[best_gt, j] >= args.froc_match_dice
                                    and not gt_matched[best_gt]):
                                is_tp[j] = True
                                gt_matched[best_gt] = True
                    froc_cands.extend(
                        (float(scores[j]), bool(is_tp[j]))
                        for j in range(n_froc_pred)
                    )

            if dump_cc:
                gt_sizes = gt_sizes_arr.tolist()
                pred_sizes = np.bincount(pr_labels.ravel(),
                                         minlength=n_pred + 1)[1:].astype(int).tolist()
                gt_best_dice = (dice_mat.max(axis=1).tolist()
                                if n_gt > 0 and n_pred > 0 else [0.0] * n_gt)
                pred_best_dice = (dice_mat.max(axis=0).tolist()
                                  if n_gt > 0 and n_pred > 0 else [0.0] * n_pred)
                ann = ann_batch
                img_name = ann["image"][b] if ann is not None else ""
                mask_name = ann["mask"][b] if ann is not None else ""
                cc_stats_slices.append({
                    "image": img_name,
                    "mask": mask_name,
                    "is_normal": mask_name == "empty",
                    "gt_sizes": gt_sizes,
                    "pred_sizes": pred_sizes,
                    "gt_best_dice": [round(float(v), 4) for v in gt_best_dice],
                    "pred_best_dice": [round(float(v), 4) for v in pred_best_dice],
                    "gt_matched": {
                        f"{thr:g}": per_thr_match[thr][0].tolist()
                        for thr in NODULE_DICE_THRS
                    },
                    "pred_matched": {
                        f"{thr:g}": per_thr_match[thr][1].tolist()
                        for thr in NODULE_DICE_THRS
                    },
                })

    if froc_cands and (args.froc_json or args.froc_plot):
        froc_cands.sort(key=lambda x: -x[0])
        tp_cum = fp_cum = 0
        fp_arr, sens_arr = [], []
        for score, is_tp in froc_cands:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            fp_arr.append(fp_cum / froc_img_total if froc_img_total > 0 else 0.0)
            sens_arr.append(tp_cum / froc_gt_total if froc_gt_total > 0 else 0.0)
        fp_arr = np.array(fp_arr)
        sens_arr = np.array(sens_arr)

        operating_points = [round(v * 0.1, 1) for v in range(1, 10)]
        sampled = {}
        for op in operating_points:
            idx = np.searchsorted(fp_arr, op, side='right') - 1
            sampled[str(op)] = float(sens_arr[idx]) if idx >= 0 else 0.0
        mean_sens = float(np.mean(list(sampled.values())))

        print(f'\nFROC ({froc_gt_total} nodules, {froc_img_total} images):')
        for op, s in sampled.items():
            print(f'  FP/img={op}: sens={s:.4f}')
        print(f'  Mean sensitivity: {mean_sens:.4f}')

        if args.froc_json:
            froc_path = Path(args.froc_json)
            froc_path.parent.mkdir(parents=True, exist_ok=True)
            with open(froc_path, "w") as f:
                json.dump({
                    "gt_nodules": froc_gt_total,
                    "images": froc_img_total,
                    "match_dice_thr": args.froc_match_dice,
                    "extract_thr": args.froc_extract_thr,
                    "operating_points": sampled,
                    "mean_sensitivity": mean_sens,
                }, f, indent=2)
            print(f'  Saved JSON → {froc_path}')

        if args.froc_plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            op_x = [float(k) for k in sampled]
            op_y = list(sampled.values())

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.step(fp_arr, sens_arr, where='post', color='steelblue',
                    linewidth=1.2, alpha=0.7, label='FROC curve')
            ax.scatter(op_x, op_y, color='steelblue', zorder=5, s=40)
            for x, y in zip(op_x, op_y):
                ax.annotate(f'{y:.2f}', (x, y),
                            textcoords='offset points', xytext=(4, 4), fontsize=7)
            ax.axhline(mean_sens, color='tomato', linestyle='--', linewidth=1,
                       label=f'Mean sens = {mean_sens:.3f}')
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 1.0)
            ax.set_xlabel('Average FP per image')
            ax.set_ylabel('Sensitivity')
            ax.set_title(
                f'FROC  |  {froc_gt_total} nodules  {froc_img_total} images\n'
                f'Dice match ≥ {args.froc_match_dice}  extract thr = {args.froc_extract_thr}'
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plot_path = Path(args.froc_plot)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved plot → {plot_path}')

    mean_iou = (torch.cat(iou_chunks).mean().item() if iou_chunks else 0.0) * 100.0
    overall_iou = ((cum_inter / cum_union).item() * 100.0) if cum_union.item() > 0 else 0.0
    tn_rate = (n_tn / n_neg) * 100.0 if n_neg > 0 else 0.0

    mean_nodule_dice = (nodule_dice_sum / nodule_dice_count * 100.0
                        if nodule_dice_count > 0 else 0.0)

    print('Final results:')
    print(f'  Mean IoU:         {mean_iou:.2f}  ({n_pos} positive samples)')
    print(f'  Overall IoU:      {overall_iou:.2f}')
    print(f'  TN rate:          {tn_rate:.2f}  ({n_tn}/{n_neg} negatives all-zero)')
    print(f'  Mean nodule Dice: {mean_nodule_dice:.2f}  ({nodule_dice_count} GT nodules, best-match)')
    for thr, ok in zip(iou_thresholds, seg_correct):
        prec = (100.0 * ok / n_pos) if n_pos > 0 else 0.0
        print(f'  precision@{thr:.1f}: {prec:.2f}')

    def _format(tag: str, table: dict) -> None:
        print(f'\nPer-nodule metrics, Dice match (tag: {tag}):')
        print(f'  {"Dice":>4} | {"TP":>5} {"FN":>5} {"FP":>5} | '
              f'{"Recall":>7} {"Precision":>9}')
        for thr in NODULE_DICE_THRS:
            tp, fn, fp = table[thr]
            recall = (100.0 * tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            precision = (100.0 * tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            print(f'  {thr:>4.2f} | {tp:>5d} {fn:>5d} {fp:>5d} | '
                  f'{recall:>6.2f}% {precision:>8.2f}%')

    _format('nodule slices only', cc_nod)
    _format('including normal slices', cc_all)
    _format(f'nodule slices only  [filtered GT≥{GT_SIZE_MIN_PX}px]', cc_nod_f)
    _format(f'including normal    [filtered GT≥{GT_SIZE_MIN_PX}px]', cc_all_f)

    print('\nSize-stratified metrics (GT bin = GT nodule area, FP bin = pred CC area):')
    print(f'  {"Size bin":<22}', end='')
    for thr in NODULE_DICE_THRS:
        print(f'  Rec≥{thr:.1f}  Pre≥{thr:.1f}', end='')
    print()
    for bi, label in enumerate(SIZE_LABELS):
        lbl = label.replace('\n', ' ')
        print(f'  {lbl:<22}', end='')
        for thr in NODULE_DICE_THRS:
            matched, total = size_recall[thr][bi]
            fp = size_fp[thr][bi]
            rec = 100.0 * matched / total if total > 0 else float('nan')
            pre = 100.0 * matched / (matched + fp) if (matched + fp) > 0 else float('nan')
            print(f'  {rec:>5.1f}%  {pre:>5.1f}%', end='')
        print(f'  (n={total})')

    if args.size_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_bins = len(SIZE_BINS)
        x = np.arange(n_bins)
        counts = [size_recall[NODULE_DICE_THRS[0]][bi][1] for bi in range(n_bins)]

        recalls    = {thr: [] for thr in NODULE_DICE_THRS}
        precisions = {thr: [] for thr in NODULE_DICE_THRS}
        for thr in NODULE_DICE_THRS:
            for bi in range(n_bins):
                matched, total = size_recall[thr][bi]
                fp = size_fp[thr][bi]
                recalls[thr].append(
                    100.0 * matched / total if total > 0 else float('nan'))
                precisions[thr].append(
                    100.0 * matched / (matched + fp) if (matched + fp) > 0 else float('nan'))

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        # Bars: one per category, coloured by clinical category
        bars = ax1.bar(x, counts, color=CAT_COLORS, alpha=0.55,
                       width=0.5, label='Nodule count', zorder=2)
        max_cnt = max(counts) if counts else 1
        for bar, cnt in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max_cnt * 0.01,
                     str(cnt), ha='center', va='bottom', fontsize=9)

        # Lines: recall (solid) and precision (dashed) for each Dice threshold
        line_styles = [
            ('#3a7ebf', '-',  'o', 'Recall'),
            ('#e05c2a', '--', 's', 'Precision'),
        ]
        for thr in NODULE_DICE_THRS:
            for (color, ls, mk, lname), vals in zip(
                    line_styles, [recalls[thr], precisions[thr]]):
                ax2.plot(x, vals, marker=mk, color=color, linestyle=ls,
                         linewidth=1.8, markersize=6,
                         label=f'{lname} Dice≥{thr}', zorder=3)
                for xi, v in zip(x, vals):
                    if not np.isnan(v):
                        offset = 7 if lname == 'Recall' else -14
                        ax2.annotate(f'{v:.1f}%', (xi, v),
                                     textcoords='offset points',
                                     xytext=(0, offset),
                                     ha='center', fontsize=7.5, color=color)

        ax1.set_xticks(x)
        ax1.set_xticklabels(SIZE_LABELS, fontsize=10)
        ax1.set_ylabel('Nodule count', fontsize=11)
        ax1.set_ylim(0, max_cnt * 1.3)
        ax2.set_ylabel('Recall / Precision (%)', fontsize=11)
        ax2.set_ylim(0, 115)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper left', fontsize=8, ncol=2)

        ax1.set_title('Nodule count & Recall / Precision by clinical size category',
                      fontsize=11)
        ax1.grid(axis='y', alpha=0.3, zorder=1)

        size_plot_path = Path(args.size_plot)
        size_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(size_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved size plot → {size_plot_path}')

    if dump_cc:
        out_path = Path(args.cc_stats_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_h, img_w = output_size
        payload = {
            "img_size": f"{img_w}x{img_h}",
            "img_h": img_h,
            "img_w": img_w,
            "match_metric": "dice",
            "thresholds": list(NODULE_DICE_THRS),
            "slices": cc_stats_slices,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f)
        print(f'\nWrote per-slice CC stats to {out_path} '
              f'({len(cc_stats_slices)} slices)')


def main(args):
    device = torch.device(args.device)

    dataset_test = LungNoduleDataset(
        data_root=args.data_root,
        split=args.split,
        transforms=get_transform(args),
        neg_ratio=args.neg_ratio,
        seed=args.seed,
        return_meta=args.save_pred or bool(args.cc_stats_json),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.pin_mem,
    )

    n_neg_in_set = len(dataset_test.samples) - len(dataset_test.positives)
    print(f'Model: {args.model}')
    print(f'Test split "{args.split}": {len(dataset_test)} samples '
          f'({len(dataset_test.positives)} pos + {n_neg_in_set} neg)')

    model = segmentation.__dict__[args.model](pretrained='', args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    evaluate(model, data_loader, device=device, args=args)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print(f'Image size: {format_input_size(args)}')
    main(args)
