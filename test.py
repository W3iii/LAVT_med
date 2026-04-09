"""
test_bc.py
──────────
Test / inference script for LAVT-BC on the BC breast-cancer dataset.
Adapted from the original test.py — uses BCDataset instead of ReferDataset.

Because BCDataset always returns ONE randomly-chosen sentence, evaluation
iterates over each sample's full sentence pool explicitly (including the
empty-string entry) to mirror the original LAVT per-sentence evaluation loop.

Example:
    python test_bc.py \
        --model lavt \
        --model_id lavt_bc \
        --bc_dataset_root ../dataset \
        --split test \
        --resume ./checkpoints/bc/model_best_lavt_bc.pth \
        --swin_type base \
        --window12 \
        --img_size 384
"""

import os
import json
import numpy as np
import torch
import torch.utils.data

from transformers import BertModel
from lib import segmentation
import transforms as T
import utils


# ── dataset ───────────────────────────────────────────────────────────────────

def get_dataset(split, transform, args):
    from data.dataset_bc import BCDataset
    ds = BCDataset(
        args,
        split=split,
        image_transforms=transform,
        target_transforms=None,
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


# ── IoU helpers ───────────────────────────────────────────────────────────────

def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))
    return I, U


def computeDice(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    denom = np.sum(pred_seg) + np.sum(gd_seg)
    return (2 * I / denom) if denom > 0 else 0.0


# ── evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model, data_loader, bert_model, device, all_sentences,
             save_pred=False, output_dir=None, dataset=None):
    """
    all_sentences : list[list[tuple(input_ids_tensor, attn_mask_tensor)]]
        Pre-tokenised sentence pool per sample (same order as data_loader).
        Each inner list has len == number of sentences + 1 (empty string).
    """
    from PIL import Image as PILImage

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if save_pred and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        print(f'Saving prediction PNGs to: {output_dir}')

    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    mean_Dice = []

    # per-sentence IoU/Dice accumulation: {sent_idx: [value, ...]}
    sent_iou_dict  = {}
    sent_dice_dict = {}

    with torch.no_grad():
        for idx, data in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image, target, _, _ = data          # sentences/attentions not used here
            image  = image.to(device)
            target = target.cpu().numpy()

            # iterate over every sentence in this sample's pool
            sent_pool = all_sentences[idx]      # list of (ids_tensor, mask_tensor)
            for sent_idx, (ids_t, mask_t) in enumerate(sent_pool):
                ids_t  = ids_t.to(device)       # (1, max_tokens)
                mask_t = mask_t.to(device)

                if bert_model is not None:
                    last_hidden = bert_model(ids_t, attention_mask=mask_t)[0]
                    embedding   = last_hidden.permute(0, 2, 1)          # (1, 768, N_l)
                    output      = model(image, embedding,
                                        l_mask=mask_t.unsqueeze(-1))
                else:
                    output = model(image, ids_t, l_mask=mask_t)

                output_mask = output.cpu().argmax(1).numpy()
                I, U = computeIoU(output_mask, target)
                this_iou  = (I / U) if U > 0 else 0.0
                this_dice = computeDice(output_mask, target)
                mean_IoU.append(this_iou)
                mean_Dice.append(this_dice)
                cum_I += I
                cum_U += U
                for n, eval_iou in enumerate(eval_seg_iou_list):
                    seg_correct[n] += (this_iou >= eval_iou)
                seg_total += 1
                sent_iou_dict.setdefault(sent_idx, []).append(this_iou)
                sent_dice_dict.setdefault(sent_idx, []).append(this_dice)

                # ── save prediction PNG ───────────────────────────────────
                if save_pred and output_dir is not None and dataset is not None:
                    ann      = dataset.annotations[idx]
                    img_stem = os.path.splitext(ann['image'])[0]

                    # load original image (grayscale) and GT mask
                    orig_img = PILImage.open(
                        os.path.join(dataset.image_dir, ann['image'])
                    ).convert('L')
                    gt_pil   = PILImage.open(
                        os.path.join(dataset.mask_dir, ann['mask'])
                    )

                    # match the model's output resolution
                    H, W = output_mask.shape[-2], output_mask.shape[-1]
                    orig_img = orig_img.resize((W, H), PILImage.BILINEAR)
                    gt_pil   = gt_pil.resize((W, H), PILImage.NEAREST)

                    orig_arr = np.array(orig_img)                        # H×W uint8
                    gt_bin   = (np.array(gt_pil) > 0)                   # H×W bool
                    pred_bin = (output_mask[0] > 0)                     # H×W bool

                    def overlay_color(gray, mask, color, alpha=0.45):
                        """Overlay a colour (R,G,B) on a grayscale array where mask==True."""
                        rgb = np.stack([gray, gray, gray], axis=-1).copy()
                        for ch, val in enumerate(color):
                            rgb[mask, ch] = np.clip(
                                gray[mask].astype(np.float32) * (1 - alpha) + val * alpha, 0, 255
                            ).astype(np.uint8)
                        return rgb

                    left  = overlay_color(orig_arr, gt_bin,   color=(0, 255, 0))   # GT   → green
                    right = overlay_color(orig_arr, pred_bin, color=(255, 0, 0))   # Pred → red

                    # 2-column: GT overlay | Pred overlay
                    canvas = np.concatenate([left, right], axis=1)
                    sent_dir = os.path.join(output_dir, f's{sent_idx}')
                    os.makedirs(sent_dir, exist_ok=True)
                    save_name = f'{img_stem}_iou{this_iou:.2f}.png'
                    PILImage.fromarray(canvas).save(
                        os.path.join(sent_dir, save_name)
                    )

            del image, target

    mean_IoU  = np.array(mean_IoU)
    mean_Dice = np.array(mean_Dice)
    mIoU  = float(np.mean(mean_IoU))
    mDice = float(np.mean(mean_Dice))
    overall_iou = cum_I / cum_U if cum_U > 0 else 0.0

    print('Final results:')
    print('  Mean IoU  : %.2f%%' % (mIoU  * 100.))
    print('  Mean Dice : %.2f%%' % (mDice * 100.))
    results_str = ''
    for n, eval_iou in enumerate(eval_seg_iou_list):
        results_str += '    precision@%.1f = %.2f%%\n' % (
            eval_iou, seg_correct[n] * 100. / seg_total
        )
    results_str += '    overall IoU = %.2f%%\n' % (overall_iou * 100.)
    print(results_str)

    # ── per-sentence breakdown ────────────────────────────────────────────
    print('Per-sentence mean IoU / Dice:')
    best_sent, best_iou = -1, -1.0
    per_sentence = {}
    for s_idx in sorted(sent_iou_dict.keys()):
        s_iou  = float(np.mean(sent_iou_dict[s_idx]))
        s_dice = float(np.mean(sent_dice_dict[s_idx]))
        print(f'  s{s_idx}: IoU={s_iou*100:.2f}%  Dice={s_dice*100:.2f}%')
        per_sentence[f's{s_idx}'] = {'mean_iou': round(s_iou, 6),
                                     'mean_dice': round(s_dice, 6)}
        if s_iou > best_iou:
            best_iou  = s_iou
            best_sent = s_idx
    print(f'\n  >> Best sentence: s{best_sent}  (mean IoU = {best_iou*100:.2f}%)')

    # ── save JSON ─────────────────────────────────────────────────────────
    results = {
        'mean_iou':    round(mIoU,  6),
        'mean_dice':   round(mDice, 6),
        'overall_iou': round(overall_iou, 6),
        'precision': {
            f'@{eval_iou}': round(seg_correct[n] / seg_total, 6)
            for n, eval_iou in enumerate(eval_seg_iou_list)
        },
        'per_sentence': per_sentence,
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
    parser = argparse.ArgumentParser(description='LAVT-BC testing')

    parser.add_argument('--bc_dataset_root', default='../dataset')
    parser.add_argument('--split', default='test',
                        help='which split to evaluate: val or test')
    parser.add_argument('--model', default='lavt')
    parser.add_argument('--model_id', default='lavt_bc')
    parser.add_argument('--swin_type', default='base')
    parser.add_argument('--window12', action='store_true')
    parser.add_argument('--mha', default='')
    parser.add_argument('--fusion_drop', default=0.0, type=float)
    parser.add_argument('--img_size', default=384, type=int)
    parser.add_argument('--resume', required=True,
                        help='path to the checkpoint to evaluate')
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased')
    parser.add_argument('--ck_bert', default='bert-base-uncased')
    parser.add_argument('--ddp_trained_weights', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--pretrained_swin_weights', default='')
    parser.add_argument('--save_pred', action='store_true',
                        help='save prediction PNGs (original | GT | pred) to output_dir')
    parser.add_argument('--output_dir', default='./pred_results',
                        help='directory to save prediction PNG files')

    return parser


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(args.device)

    dataset_test, _ = get_dataset(args.split, get_transform(args), args)
    test_sampler     = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
    )

    print(f'Testing on split "{args.split}": {len(dataset_test)} samples')

    # build per-sample sentence pools: list of (ids, mask) pairs
    # mirrors dataset_test.input_ids / attention_masks directly
    all_sentences = [
        list(zip(dataset_test.input_ids[i], dataset_test.attention_masks[i]))
        for i in range(len(dataset_test))
    ]

    # ── model ─────────────────────────────────────────────────────────────
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='', args=args)
    checkpoint   = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)

    # ── BERT ──────────────────────────────────────────────────────────────
    if args.model != 'lavt_one':
        single_bert_model = BertModel.from_pretrained(args.ck_bert)
        single_bert_model.pooler = None  # match train_bc.py: pooler is always removed during training
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    evaluate(model, data_loader_test, bert_model, device=device,
             all_sentences=all_sentences,
             save_pred=args.save_pred, output_dir=args.output_dir,
             dataset=dataset_test)


if __name__ == '__main__':
    parser = get_parser()
    args   = parser.parse_args()
    print(f'Image size: {args.img_size}')
    main(args)
