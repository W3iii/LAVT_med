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

NODULE_IOU_THRS = (0.3, 0.5)


def get_transform(args):
    return T.Compose([
        T.Resize(args.img_size, args.img_size),
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


def _per_object_iou_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray,
                           n_gt: int, n_pred: int) -> np.ndarray:
    """IoU between every GT component and every pred component."""
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float64)
    flat = gt_labels.ravel().astype(np.int64) * (n_pred + 1) \
        + pred_labels.ravel().astype(np.int64)
    bins = np.bincount(flat, minlength=(n_gt + 1) * (n_pred + 1))
    inter = bins.reshape(n_gt + 1, n_pred + 1)[1:, 1:].astype(np.float64)
    gt_sizes = np.bincount(gt_labels.ravel(), minlength=n_gt + 1)[1:].astype(np.float64)
    pred_sizes = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)[1:].astype(np.float64)
    union = gt_sizes[:, None] + pred_sizes[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _greedy_match(iou_mat: np.ndarray, thr: float) -> tuple:
    """Match GT/pred pairs greedily by descending IoU at threshold thr."""
    n_gt, n_pred = iou_mat.shape
    if n_gt == 0:
        return 0, 0, n_pred
    if n_pred == 0:
        return 0, n_gt, 0
    pairs_i, pairs_j = np.where(iou_mat >= thr)
    if pairs_i.size == 0:
        return 0, n_gt, n_pred
    order = np.argsort(-iou_mat[pairs_i, pairs_j])
    pairs_i, pairs_j = pairs_i[order], pairs_j[order]
    matched_gt = np.zeros(n_gt, dtype=bool)
    matched_pred = np.zeros(n_pred, dtype=bool)
    for i, j in zip(pairs_i, pairs_j):
        if not matched_gt[i] and not matched_pred[j]:
            matched_gt[i] = True
            matched_pred[j] = True
    tp = int(matched_gt.sum())
    return tp, n_gt - tp, n_pred - int(matched_pred.sum())


def _center_paste(canvas: np.ndarray, src: np.ndarray) -> None:
    """Center-paste src into canvas in-place. Crops src if it exceeds canvas."""
    ch, cw = canvas.shape[:2]
    sh, sw = src.shape[:2]
    h_use = min(sh, ch)
    w_use = min(sw, cw)
    top = (ch - h_use) // 2
    left = (cw - w_use) // 2
    src_top = (sh - h_use) // 2
    src_left = (sw - w_use) // 2
    canvas[top:top + h_use, left:left + w_use] = \
        src[src_top:src_top + h_use, src_left:src_left + w_use]


def _save_overlay(image_path: Path, mask_path, pred: torch.Tensor,
                  out_path: Path, canvas_size: int) -> None:
    """
    Restore the original aspect ratio: scale the longest side to
    canvas_size, then center-pad the shorter side with zeros. GT (green)
    and pred (red) 1-pixel contours are drawn on the rescaled CT slice.
    """
    img = Image.open(image_path).convert("L")
    W, H = img.size

    scale = canvas_size / max(W, H)
    new_W = max(1, round(W * scale))
    new_H = max(1, round(H * scale))

    img_np = np.array(img.resize((new_W, new_H), Image.BILINEAR), dtype=np.uint8)

    pred_np = pred.cpu().numpy().astype(np.uint8)
    pred_resized = np.array(
        Image.fromarray(pred_np).resize((new_W, new_H), Image.NEAREST),
        dtype=np.uint8,
    )

    rgb = np.stack([img_np, img_np, img_np], axis=-1)

    if mask_path is not None:
        gt = Image.open(mask_path)
        gt_resized = np.array(
            gt.resize((new_W, new_H), Image.NEAREST), dtype=np.uint8
        )
        rgb[_mask_outline(gt_resized)] = (0, 255, 0)

    rgb[_mask_outline(pred_resized)] = (255, 0, 0)

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    _center_paste(canvas, rgb)
    Image.fromarray(canvas).save(out_path)


@torch.no_grad()
def evaluate(model, data_loader, device, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    pred_root = None
    if args.save_pred:
        pred_root = Path(args.pred_dir)
        (pred_root / "nodule").mkdir(parents=True, exist_ok=True)
        (pred_root / "normal").mkdir(parents=True, exist_ok=True)
        print(f'Saving overlays to {pred_root}/{{nodule,normal}}')

    images_dir = Path(args.data_root) / "images" / args.split
    masks_dir = Path(args.data_root) / "masks" / args.split

    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    seg_correct = np.zeros(len(iou_thresholds), dtype=np.int64)

    iou_pos_list = []
    cum_inter = 0
    cum_union = 0
    n_pos, n_neg, n_tn = 0, 0, 0

    # Per-nodule (connected-component) tallies at each IoU threshold,
    # tracked separately for nodule-only slices vs all slices.
    cc_all = {thr: [0, 0, 0] for thr in NODULE_IOU_THRS}     # tp, fn, fp
    cc_nod = {thr: [0, 0, 0] for thr in NODULE_IOU_THRS}

    for image, target, ann_batch in metric_logger.log_every(
            data_loader, 100, header):
        image = image.to(device, non_blocking=True)

        logits = model(image)
        pred = logits.argmax(dim=1)                            # (B, h, w) at img_size
        pred_cpu = pred.cpu().numpy().astype(np.uint8)

        # Metrics in original resolution: resize pred back to native (W, H)
        # and compare against the on-disk GT mask at native size.
        for b in range(pred_cpu.shape[0]):
            img_name = ann_batch["image"][b]
            mask_name = ann_batch["mask"][b]
            is_normal = (mask_name == "empty")
            img_path = images_dir / img_name

            with Image.open(img_path) as im:
                W_orig, H_orig = im.size

            if is_normal:
                gt_orig = np.zeros((H_orig, W_orig), dtype=np.uint8)
            else:
                gt_orig = np.array(Image.open(masks_dir / mask_name),
                                   dtype=np.uint8)
                gt_orig = (gt_orig > 0).astype(np.uint8)

            pred_orig = np.array(
                Image.fromarray(pred_cpu[b]).resize(
                    (W_orig, H_orig), Image.NEAREST),
                dtype=np.uint8,
            )

            inter = int(np.logical_and(pred_orig, gt_orig).sum())
            gt_sum = int(gt_orig.sum())
            pred_sum = int(pred_orig.sum())
            union = pred_sum + gt_sum - inter

            if gt_sum > 0:
                iou = (inter / union) if union > 0 else 0.0
                iou_pos_list.append(iou)
                cum_inter += inter
                cum_union += union
                n_pos += 1
                for k, thr in enumerate(iou_thresholds):
                    if iou >= thr:
                        seg_correct[k] += 1
            else:
                n_neg += 1
                if pred_sum == 0:
                    n_tn += 1

            gt_labels, n_gt = cc_label(gt_orig)
            pr_labels, n_pred = cc_label(pred_orig)
            iou_mat = _per_object_iou_matrix(gt_labels, pr_labels, n_gt, n_pred)
            for thr in NODULE_IOU_THRS:
                tp, fn, fp = _greedy_match(iou_mat, thr)
                cc_all[thr][0] += tp
                cc_all[thr][1] += fn
                cc_all[thr][2] += fp
                if n_gt > 0:
                    cc_nod[thr][0] += tp
                    cc_nod[thr][1] += fn
                    cc_nod[thr][2] += fp

            if args.save_pred:
                subdir = "normal" if is_normal else "nodule"
                stem = os.path.splitext(img_name)[0]
                out_path = pred_root / subdir / f"{stem}_overlay.png"
                gt_path = None if is_normal else (masks_dir / mask_name)
                _save_overlay(img_path, gt_path, pred[b],
                              out_path, args.img_size)

    mean_iou = (float(np.mean(iou_pos_list)) * 100.0) if iou_pos_list else 0.0
    overall_iou = (cum_inter / cum_union * 100.0) if cum_union > 0 else 0.0
    tn_rate = (n_tn / n_neg) * 100.0 if n_neg > 0 else 0.0

    print('Final results:')
    print(f'  Mean IoU:    {mean_iou:.2f}  ({n_pos} positive samples)')
    print(f'  Overall IoU: {overall_iou:.2f}')
    print(f'  TN rate:     {tn_rate:.2f}  ({n_tn}/{n_neg} negatives all-zero)')
    for thr, ok in zip(iou_thresholds, seg_correct):
        prec = (100.0 * ok / n_pos) if n_pos > 0 else 0.0
        print(f'  precision@{thr:.1f}: {prec:.2f}')

    def _format(tag: str, table: dict) -> None:
        print(f'\nPer-nodule metrics ({tag}):')
        print(f'  {"thr":>4} | {"TP":>5} {"FN":>5} {"FP":>5} | '
              f'{"Recall":>7} {"Precision":>9}')
        for thr in NODULE_IOU_THRS:
            tp, fn, fp = table[thr]
            recall = (100.0 * tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            precision = (100.0 * tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            print(f'  {thr:>4.2f} | {tp:>5d} {fn:>5d} {fp:>5d} | '
                  f'{recall:>6.2f}% {precision:>8.2f}%')

    _format('nodule slices only', cc_nod)
    _format('including normal slices', cc_all)


def main(args):
    device = torch.device(args.device)

    dataset_test = LungNoduleDataset(
        data_root=args.data_root,
        split=args.split,
        transforms=get_transform(args),
        neg_ratio=args.neg_ratio,
        seed=args.seed,
        return_meta=True,  # always — metrics now run at the original PNG resolution
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
    print(f'Image size: {args.img_size}')
    main(args)
