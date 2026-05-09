"""
test_ln.py
──────────
Test / inference script for LAVT-LN on the Lung Nodule dataset.

Example:
    python test.py \
        --model lavt \
        --model_id lavt_ln \
        --ln_dataset_root ../dataset \
        --split test \
        --resume ./checkpoints/ln/model_best_lavt_ln.pth \
        --swin_type base \
        --window12 \
        --img_size 384
"""

import os
import json
import numpy as np
import torch
import torch.utils.data
from scipy.ndimage import label as cc_label, binary_erosion

from transformers import BertModel
from lib import segmentation
import transforms as T
import utils


# ── per-nodule (per-connected-component) evaluation ───────────────────────────

PER_NODULE_DICE_THRESHOLDS = [0.1, 0.3, 0.5]
PER_NODULE_MIN_CC_PIXELS   = 0   # set >0 to suppress tiny noise CCs


def _label_ccs(binary_mask, min_size=0):
    """Return list of boolean masks for each connected component."""
    if binary_mask.sum() == 0:
        return []
    lab, n = cc_label(binary_mask > 0)
    ccs = []
    for i in range(1, n + 1):
        m = (lab == i)
        if min_size > 0 and m.sum() < min_size:
            continue
        ccs.append(m)
    return ccs


def per_nodule_match(pred_mask_2d, gt_mask_2d, dice_threshold,
                     min_size=0):
    """
    Greedy per-CC matching by Dice. Returns (tp, fp, fn, n_pred, n_gt).
      tp: #pred CCs matched to a GT CC with Dice >= threshold
      fp: #pred CCs unmatched (incl. all pred CCs on negative slices)
      fn: #GT CCs unmatched
    """
    pred_ccs = _label_ccs(pred_mask_2d, min_size)
    gt_ccs   = _label_ccs(gt_mask_2d,   min_size)
    n_pred = len(pred_ccs)
    n_gt   = len(gt_ccs)

    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, n_gt, n_pred, n_gt

    pred_sizes = [int(c.sum()) for c in pred_ccs]
    gt_sizes   = [int(c.sum()) for c in gt_ccs]

    pairs = []   # (dice, i_pred, j_gt)
    for i, p in enumerate(pred_ccs):
        for j, g in enumerate(gt_ccs):
            inter = int(np.logical_and(p, g).sum())
            denom = pred_sizes[i] + gt_sizes[j]
            d = (2.0 * inter / denom) if denom > 0 else 0.0
            if d >= dice_threshold:
                pairs.append((d, i, j))

    pairs.sort(key=lambda x: -x[0])
    matched_p, matched_g = set(), set()
    for d, i, j in pairs:
        if i in matched_p or j in matched_g:
            continue
        matched_p.add(i)
        matched_g.add(j)

    tp = len(matched_p)
    fp = n_pred - tp
    fn = n_gt - len(matched_g)
    return tp, fp, fn, n_pred, n_gt


# ── dataset ───────────────────────────────────────────────────────────────────

def get_dataset(split, transform, args):
    from data.dataset_ln import LNDataset
    ds = LNDataset(
        args,
        split=split,
        image_transforms=transform,
        target_transforms=None,
        eval_mode=True,
    )
    return ds, 2


# ── transforms ────────────────────────────────────────────────────────────────

def get_transform(args):
    return T.Compose([
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ── IoU / Dice helpers ───────────────────────────────────────────────────────

def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))
    return I, U


def computeDice(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    denom = np.sum(pred_seg) + np.sum(gd_seg)
    return (2 * I / denom) if denom > 0 else 0.0


# ── evaluate ──────────────────────────────────────────────────────────────────

def _mask_boundary(mask, thickness=1):
    """Extract boundary pixels of a binary mask (medical contour style)."""
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, iterations=thickness)
    return mask & ~eroded


def _save_overlay(dataset, ann, output_mask, save_dir, save_name,
                  contour_thickness=1):
    """Single-image overlay: GT contour (green) + Pred contour (red) on CT."""
    from PIL import Image as PILImage

    orig_img = PILImage.open(
        os.path.join(dataset.image_dir, ann['image'])
    ).convert('L')

    if ann.get('mask') and ann['is_pos'] == 1:
        gt_pil = PILImage.open(
            os.path.join(dataset.mask_dir, ann['mask'])
        )
    else:
        w, h = orig_img.size
        gt_pil = PILImage.fromarray(np.zeros((h, w), dtype=np.uint8))

    H, W = output_mask.shape[-2], output_mask.shape[-1]
    orig_img = orig_img.resize((W, H), PILImage.BILINEAR)
    gt_pil   = gt_pil.resize((W, H), PILImage.NEAREST)

    orig_arr = np.array(orig_img)
    gt_bin   = (np.array(gt_pil) > 0)
    pred_bin = (output_mask[0] > 0)

    gt_edge   = _mask_boundary(gt_bin,   thickness=contour_thickness)
    pred_edge = _mask_boundary(pred_bin, thickness=contour_thickness)

    canvas = np.stack([orig_arr, orig_arr, orig_arr], axis=-1).copy()
    canvas[gt_edge]   = [0, 255, 0]   # GT  → green
    canvas[pred_edge] = [255, 0, 0]   # Pred → red (overwrites GT on overlap)

    PILImage.fromarray(canvas).save(os.path.join(save_dir, save_name))


def evaluate(model, data_loader, bert_model, device, dataset,
             save_pred=False, output_dir=None):

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if save_pred and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f'Saving prediction PNGs to: {output_dir}')

    # ── seg metrics (positive only) ──────────────────────────────────────
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    mean_Dice = []
    cat_iou  = {}
    cat_dice = {}

    neg_total = 0
    neg_correct_seg = 0      # TN by seg head (pred all zero)

    # ── per-nodule (per-CC) stats ─────────────────────────────────────────
    # accumulators[thr] = {'tp':0,'fp':0,'fn':0,'n_pred':0,'n_gt':0}
    nod_acc     = {t: {'tp':0,'fp':0,'fn':0,'n_pred':0,'n_gt':0}
                   for t in PER_NODULE_DICE_THRESHOLDS}
    # per category (only populated by category present on slice)
    nod_acc_cat = {t: {} for t in PER_NODULE_DICE_THRESHOLDS}

    with torch.no_grad():
        for idx, data in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image, target, sentences, attentions, meta = data
            image      = image.to(device)
            target_np  = target.cpu().numpy()
            sentences  = sentences.to(device).squeeze(1)
            attentions = attentions.to(device).squeeze(1)
            category   = meta['category'].to(device)

            is_pos = meta['is_pos'].item()
            cat    = meta['category'].item()

            # ── forward ──────────────────────────────────────────────────
            if bert_model is not None:
                last_hidden = bert_model(sentences, attention_mask=attentions)[0]
                embedding   = last_hidden.permute(0, 2, 1)
                attentions  = attentions.unsqueeze(dim=-1)
                seg_out, exist_out = model(
                    image, embedding, l_mask=attentions, category=category)
            else:
                seg_out, exist_out = model(
                    image, sentences, l_mask=attentions, category=category)

            exist_prob = torch.sigmoid(exist_out).item()
            output_mask = seg_out.cpu().argmax(1).numpy()

            # ── positive sample: compute seg metrics ─────────────────────
            if is_pos == 1:
                I, U = computeIoU(output_mask, target_np)
                this_iou  = (I / U) if U > 0 else 0.0
                this_dice = computeDice(output_mask, target_np)
                mean_IoU.append(this_iou)
                mean_Dice.append(this_dice)
                cum_I += I
                cum_U += U
                for n, eval_iou in enumerate(eval_seg_iou_list):
                    seg_correct[n] += (this_iou >= eval_iou)
                seg_total += 1

                cat_iou.setdefault(cat, []).append(this_iou)
                cat_dice.setdefault(cat, []).append(this_dice)

                # ── per-nodule (per-CC) ─────────────────────────────────
                pred2d = output_mask[0]
                gt2d   = target_np[0]
                for thr in PER_NODULE_DICE_THRESHOLDS:
                    tp, fp, fn, n_p, n_g = per_nodule_match(
                        pred2d, gt2d, thr, PER_NODULE_MIN_CC_PIXELS)
                    a = nod_acc[thr]
                    a['tp'] += tp; a['fp'] += fp; a['fn'] += fn
                    a['n_pred'] += n_p; a['n_gt'] += n_g
                    ca = nod_acc_cat[thr].setdefault(
                        cat, {'tp':0,'fp':0,'fn':0,'n_pred':0,'n_gt':0})
                    ca['tp'] += tp; ca['fp'] += fp; ca['fn'] += fn
                    ca['n_pred'] += n_p; ca['n_gt'] += n_g

                # ── save prediction PNG ──────────────────────────────────
                if save_pred and output_dir is not None:
                    ann = dataset.annotations[idx]
                    img_stem = os.path.splitext(ann['image'])[0]
                    save_dir = os.path.join(output_dir, 'pos', f'cls{cat}')
                    os.makedirs(save_dir, exist_ok=True)
                    save_name = f'{img_stem}_iou{this_iou:.2f}.png'
                    _save_overlay(dataset, ann, output_mask, save_dir, save_name)
            else:
                # ── negative sample ──────────────────────────────────────
                neg_total += 1
                has_fp = output_mask.sum() > 0
                if not has_fp:
                    neg_correct_seg += 1

                # ── per-nodule: any pred CC counts as FP ────────────────
                pred2d = output_mask[0]
                gt2d   = np.zeros_like(pred2d)
                for thr in PER_NODULE_DICE_THRESHOLDS:
                    tp, fp, fn, n_p, n_g = per_nodule_match(
                        pred2d, gt2d, thr, PER_NODULE_MIN_CC_PIXELS)
                    a = nod_acc[thr]
                    a['tp'] += tp; a['fp'] += fp; a['fn'] += fn
                    a['n_pred'] += n_p; a['n_gt'] += n_g

                # ── save prediction PNG ──────────────────────────────────
                if save_pred and output_dir is not None:
                    ann = dataset.annotations[idx]
                    img_stem = os.path.splitext(ann['image'])[0]
                    sub = 'fp' if has_fp else 'tn'
                    save_dir = os.path.join(output_dir, 'neg', sub)
                    os.makedirs(save_dir, exist_ok=True)
                    save_name = f'{img_stem}_ep{exist_prob:.2f}.png'
                    _save_overlay(dataset, ann, output_mask, save_dir, save_name)

            del image

    # ── summary ──────────────────────────────────────────────────────────────
    mean_IoU  = np.array(mean_IoU)
    mean_Dice = np.array(mean_Dice)
    mIoU  = float(np.mean(mean_IoU))  if len(mean_IoU) > 0 else 0.0
    mDice = float(np.mean(mean_Dice)) if len(mean_Dice) > 0 else 0.0
    overall_iou = cum_I / cum_U if cum_U > 0 else 0.0

    tn_rate_seg = neg_correct_seg / neg_total if neg_total > 0 else 0.0

    print('\n' + '=' * 60)
    print('Final results:')
    print(f'  Positive samples: {seg_total}')
    print(f'  Mean IoU  : {mIoU*100:.2f}%')
    print(f'  Mean Dice : {mDice*100:.2f}%')
    results_str = ''
    for n, eval_iou in enumerate(eval_seg_iou_list):
        results_str += '    precision@%.1f = %.2f%%\n' % (
            eval_iou, seg_correct[n] * 100. / max(seg_total, 1)
        )
    results_str += f'    overall IoU = {overall_iou*100:.2f}%\n'
    results_str += f'  Negative samples: {neg_total}\n'
    results_str += f'    TN rate (seg head): {tn_rate_seg*100:.2f}%\n'
    print(results_str)

    # ── per-category breakdown ────────────────────────────────────────────
    cat_names = {0: 'normal', 1: 'benign', 2: 'prob_benign',
                 3: 'prob_suspicious', 4: 'suspicious'}
    print('Per-category mean IoU / Dice:')
    per_category = {}
    for cat in sorted(cat_iou.keys()):
        c_iou  = float(np.mean(cat_iou[cat]))
        c_dice = float(np.mean(cat_dice[cat]))
        name   = cat_names.get(cat, f'cat{cat}')
        print(f'  {name} (cls{cat}): IoU={c_iou*100:.2f}%  Dice={c_dice*100:.2f}%  '
              f'(n={len(cat_iou[cat])})')
        per_category[name] = {'mean_iou': round(c_iou, 6),
                              'mean_dice': round(c_dice, 6),
                              'count': len(cat_iou[cat])}

    # ── per-nodule (per-CC) summary ───────────────────────────────────────
    def _stats(d):
        tp, fp, fn = d['tp'], d['fp'], d['fn']
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2*prec*recall / (prec+recall) if (prec+recall) > 0 else 0.0
        return prec, recall, f1

    print('\nPer-nodule (per-CC) results:')
    per_nodule = {}
    for thr in PER_NODULE_DICE_THRESHOLDS:
        a = nod_acc[thr]
        prec, recall, f1 = _stats(a)
        print(f'  Dice >= {thr}:  TP={a["tp"]}  FP={a["fp"]}  FN={a["fn"]}  '
              f'(GT={a["n_gt"]}, Pred={a["n_pred"]})')
        print(f'    Precision={prec*100:.2f}%  Recall={recall*100:.2f}%  '
              f'F1={f1*100:.2f}%')
        thr_key = f'@{thr}'
        per_nodule[thr_key] = {
            'tp': a['tp'], 'fp': a['fp'], 'fn': a['fn'],
            'n_pred': a['n_pred'], 'n_gt': a['n_gt'],
            'precision': round(prec, 6),
            'recall':    round(recall, 6),
            'f1':        round(f1, 6),
            'per_category': {},
        }
        for cat in sorted(nod_acc_cat[thr].keys()):
            ca = nod_acc_cat[thr][cat]
            cp, cr, cf = _stats(ca)
            name = cat_names.get(cat, f'cat{cat}')
            print(f'    [{name}] TP={ca["tp"]} FP={ca["fp"]} FN={ca["fn"]}  '
                  f'P={cp*100:.2f}% R={cr*100:.2f}% F1={cf*100:.2f}%')
            per_nodule[thr_key]['per_category'][name] = {
                'tp': ca['tp'], 'fp': ca['fp'], 'fn': ca['fn'],
                'n_pred': ca['n_pred'], 'n_gt': ca['n_gt'],
                'precision': round(cp, 6),
                'recall':    round(cr, 6),
                'f1':        round(cf, 6),
            }

    # ── save JSON ─────────────────────────────────────────────────────────
    results = {
        'mean_iou':    round(mIoU,  6),
        'mean_dice':   round(mDice, 6),
        'overall_iou': round(overall_iou, 6),
        'tn_rate_seg':  round(tn_rate_seg, 6),
        'positive_count': seg_total,
        'negative_count': neg_total,
        'precision': {
            f'@{eval_iou}': round(seg_correct[n] / max(seg_total, 1), 6)
            for n, eval_iou in enumerate(eval_seg_iou_list)
        },
        'per_category': per_category,
        'per_nodule': per_nodule,
    }
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, 'results.json')
    else:
        json_path = 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {json_path}')


# ── argument parsing ──────────────────────────────────────────────────────────

def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='LAVT-LN testing')

    parser.add_argument('--ln_dataset_root', default='../dataset')
    parser.add_argument('--split', default='test',
                        help='which split to evaluate: val or test')
    parser.add_argument('--model', default='lavt')
    parser.add_argument('--model_id', default='lavt_ln')
    parser.add_argument('--swin_type', default='base')
    parser.add_argument('--window12', action='store_true')
    parser.add_argument('--mha', default='')
    parser.add_argument('--fusion_drop', default=0.0, type=float)
    parser.add_argument('--img_size', default=384, type=int)
    parser.add_argument('--resume', required=True,
                        help='path to the checkpoint to evaluate')
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased')
    parser.add_argument('--ck_bert', default='bert-base-uncased')
    parser.add_argument('--neg_ratio', default=2.0, type=float)
    parser.add_argument('--ddp_trained_weights', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--pretrained_swin_weights', default='')
    parser.add_argument('--save_pred', action='store_true',
                        help='save prediction PNGs (GT | pred overlay)')
    parser.add_argument('--output_dir', default='./pred_results_ln',
                        help='directory to save prediction PNG and results')
    parser.add_argument('--pos_only', action='store_true',
                        help='evaluate on positive samples only (is_pos==1)')

    return parser


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(args.device)

    dataset_test, _ = get_dataset(args.split, get_transform(args), args)
    if args.pos_only:
        before = len(dataset_test.annotations)
        dataset_test.annotations = [
            a for a in dataset_test.annotations if a['is_pos'] == 1]
        print(f'  --pos_only: filtered {before} -> {len(dataset_test.annotations)} '
              f'(positive samples only)')
    test_sampler     = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
    )

    print(f'Testing on split "{args.split}": {len(dataset_test)} samples')

    # ── model ─────────────────────────────────────────────────────────────
    print(f'Model: {args.model}')
    single_model = segmentation.__dict__[args.model](pretrained='', args=args)
    checkpoint   = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)

    # ── BERT ──────────────────────────────────────────────────────────────
    if args.model != 'lavt_one':
        single_bert_model = BertModel.from_pretrained(args.ck_bert)
        single_bert_model.pooler = None
        if 'bert_model' in checkpoint:
            single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    evaluate(model, data_loader_test, bert_model, device=device,
             dataset=dataset_test,
             save_pred=args.save_pred, output_dir=args.output_dir)


if __name__ == '__main__':
    parser = get_parser()
    args   = parser.parse_args()
    print(f'Image size: {args.img_size}')
    main(args)
