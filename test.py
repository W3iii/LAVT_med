import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image

from lib import segmentation
from data.dataset_lung_nodule import LungNoduleDataset
import transforms as T
import utils


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

    if args.save_pred:
        pred_root = Path(args.pred_dir)
        (pred_root / "nodule").mkdir(parents=True, exist_ok=True)
        (pred_root / "normal").mkdir(parents=True, exist_ok=True)
        images_dir = Path(args.data_root) / "images" / args.split
        masks_dir = Path(args.data_root) / "masks" / args.split
        print(f'Saving overlays to {pred_root}/{{nodule,normal}}')

    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    seg_correct = np.zeros(len(iou_thresholds), dtype=np.int64)

    iou_chunks = []
    cum_inter = torch.zeros((), dtype=torch.float64, device=device)
    cum_union = torch.zeros((), dtype=torch.float64, device=device)
    n_pos, n_neg, n_tn = 0, 0, 0

    for batch in metric_logger.log_every(data_loader, 100, header):
        if args.save_pred:
            image, target, ann_batch = batch
        else:
            image, target = batch
            ann_batch = None

        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(image)
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
                              pred[i], out_path, args.img_size)

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

    mean_iou = (torch.cat(iou_chunks).mean().item() if iou_chunks else 0.0) * 100.0
    overall_iou = ((cum_inter / cum_union).item() * 100.0) if cum_union.item() > 0 else 0.0
    tn_rate = (n_tn / n_neg) * 100.0 if n_neg > 0 else 0.0

    print('Final results:')
    print(f'  Mean IoU:    {mean_iou:.2f}  ({n_pos} positive samples)')
    print(f'  Overall IoU: {overall_iou:.2f}')
    print(f'  TN rate:     {tn_rate:.2f}  ({n_tn}/{n_neg} negatives all-zero)')
    for thr, ok in zip(iou_thresholds, seg_correct):
        prec = (100.0 * ok / n_pos) if n_pos > 0 else 0.0
        print(f'  precision@{thr:.1f}: {prec:.2f}')


def main(args):
    device = torch.device(args.device)

    dataset_test = LungNoduleDataset(
        data_root=args.data_root,
        split=args.split,
        transforms=get_transform(args),
        neg_ratio=args.neg_ratio,
        seed=args.seed,
        return_meta=args.save_pred,
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
