"""
train_ln.py  (fixed v3)
───────────
Fix: removed all lazy-build logic for exist_head.
exist_head params are in params_to_optimize from the start.
"""

import argparse
import datetime
import os
import time
from functools import reduce
import operator
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import nn

import torchvision
from collections import defaultdict

import transforms as T
import utils
from transformers import BertModel
from lib import segmentation


# ── argument parsing ──────────────────────────────────────────────────────────

def get_parser():
    parser = argparse.ArgumentParser(description='LAVT-LN training')

    parser.add_argument('--ln_dataset_root', default='../dataset')
    parser.add_argument('--model', default='lavt')
    parser.add_argument('--model_id', default='lavt_ln')
    parser.add_argument('--swin_type', default='base')
    parser.add_argument('--pretrained_swin_weights', default='')
    parser.add_argument('--window12', action='store_true')
    parser.add_argument('--mha', default='')
    parser.add_argument('--fusion_drop', default=0.0, type=float)
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased')
    parser.add_argument('--ck_bert', default='bert-base-uncased')
    parser.add_argument('--neg_ratio', default=2.0, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--img_size', default=384, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', dest='weight_decay')
    parser.add_argument('--amsgrad', action='store_true')
    parser.add_argument('--early_stop', default=10, type=int)
    parser.add_argument('--output-dir', default='./checkpoints/ln/')
    parser.add_argument('--resume', default='')
    parser.add_argument('--ddp_trained_weights', action='store_true')
    parser.add_argument('--print-freq', default=10, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--patch_size', default=0, type=int,
                        help='Foreground crop patch size (0=disabled). e.g. 128')
    parser.add_argument('--fg_prob', default=0.67, type=float,
                        help='Probability of foreground-centered crop (default 0.67)')
    parser.add_argument('--iters_per_epoch', default=0, type=int,
                        help='nnU-Net style: fixed iterations per epoch (0=traditional epoch)')
    parser.add_argument('--val_every', default=5, type=int,
                        help='Run validation every N epochs (default 5)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--local_rank', type=int, default=-1)

    return parser


# ── patient-aware batch sampler ────────────────────────────────────────────────

class PatientAwareBatchSampler(torch.utils.data.Sampler):
    def __init__(self, annotations, batch_size, drop_last=True, seed=42):
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.seed       = seed
        self.epoch      = 0
        patient_indices = defaultdict(list)
        for idx, ann in enumerate(annotations):
            patient_indices[ann['patient_id']].append(idx)
        self.patient_indices = dict(patient_indices)
        self.total = sum(len(v) for v in self.patient_indices.values())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        queues = {}
        for pid, indices in self.patient_indices.items():
            q = indices[:]
            rng.shuffle(q)
            queues[pid] = q
        pids = list(queues.keys())
        flat = []
        max_len = max(len(v) for v in queues.values())
        for i in range(max_len):
            rng.shuffle(pids)
            for pid in pids:
                if i < len(queues[pid]):
                    flat.append(queues[pid][i])
        for start in range(0, len(flat), self.batch_size):
            batch = flat[start:start + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last:
                yield batch

    def __len__(self):
        if self.drop_last:
            return self.total // self.batch_size
        return (self.total + self.batch_size - 1) // self.batch_size


def get_dataset(split, transform, args):
    from data.dataset_ln import LNDataset
    ds = LNDataset(
        args, split=split, image_transforms=transform,
        target_transforms=None, eval_mode=False)
    num_classes = 2
    return ds, num_classes


def get_transform(args):
    patch_size = getattr(args, 'patch_size', 0)
    fg_prob = getattr(args, 'fg_prob', 0.67)

    tfms = []
    if patch_size > 0:
        # Foreground oversampling: crop patch first, then resize to model input
        tfms.append(T.ForegroundCrop(patch_size, fg_prob=fg_prob))
    tfms += [
        T.Resize(args.img_size, args.img_size),
        T.RandomHorizontalFlip(flip_prob=0.5),
        T.RandomAffine(angle=(-15, 15), translate=(0.10, 0.10),
                       scale=(0.85, 1.15), shear=(-5, 5)),
        T.ToTensor(),
        T.RandomGaussianNoise(std=0.02, p=0.5),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(tfms)


# ── per-category loss weight ───────────────────────────────────────────────────
CLS_LOSS_WEIGHT = {0: 1.0, 1: 2.0, 2: 1.5, 3: 1.0, 4: 1.0}
_WEIGHT_TABLE = torch.tensor(
    [CLS_LOSS_WEIGHT.get(i, 1.0) for i in range(5)], dtype=torch.float32)


def dice_loss_per_sample(input, target, smooth=1.0):
    prob         = torch.softmax(input, dim=1)[:, 1]
    gt           = target.float()
    intersection = (prob * gt).sum(dim=(1, 2))
    union        = prob.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
    dice         = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def focal_loss_per_sample(input, target, alpha=0.75, gamma=2.0):
    ce      = F.cross_entropy(input, target, reduction='none')
    prob    = torch.softmax(input, dim=1)
    p_t     = prob.gather(1, target.unsqueeze(1)).squeeze(1)
    alpha_t = torch.where(
        target == 1,
        torch.tensor(alpha,       device=input.device),
        torch.tensor(1.0 - alpha, device=input.device))
    focal_w = alpha_t * (1.0 - p_t) ** gamma
    return (focal_w * ce).mean(dim=(1, 2))


def criterion(seg_out, exist_out, target, is_pos,
              category=None, exist_weight=1.0):
    exist_gt   = is_pos.float()
    exist_loss = F.binary_cross_entropy_with_logits(
        exist_out, exist_gt,
        pos_weight=torch.tensor(6.0, device=seg_out.device))

    has_fg = is_pos.bool()

    if has_fg.any():
        pos_input  = seg_out[has_fg]
        pos_target = target[has_fg]
        dice_per  = dice_loss_per_sample(pos_input, pos_target)
        focal_per = focal_loss_per_sample(pos_input, pos_target)
        loss_per  = focal_per + dice_per
        if category is not None:
            pos_cat = category[has_fg]
            cls_w   = _WEIGHT_TABLE.to(seg_out.device)[pos_cat]
            pos_seg_loss = (loss_per * cls_w).sum() / cls_w.sum()
        else:
            pos_seg_loss = loss_per.mean()
    else:
        pos_seg_loss = torch.tensor(0.0, device=seg_out.device)

    has_bg = ~has_fg
    if has_bg.any():
        neg_input  = seg_out[has_bg]
        neg_target = target[has_bg]
        neg_focal = focal_loss_per_sample(neg_input, neg_target)
        neg_seg_loss = neg_focal.mean()
    else:
        neg_seg_loss = torch.tensor(0.0, device=seg_out.device)

    seg_loss = pos_seg_loss + 0.5 * neg_seg_loss
    return seg_loss + exist_weight * exist_loss


def class_embed_contrastive_loss(model):
    embs = model.class_embed.weight[1:5]
    embs = F.normalize(embs, dim=-1)
    sim = torch.mm(embs, embs.t())
    mask = ~torch.eye(4, dtype=torch.bool, device=sim.device)
    return sim[mask].pow(2).mean()


def IoU(pred, gt):
    pred = pred.argmax(1)
    intersection = torch.sum(torch.mul(pred, gt))
    union        = torch.sum(torch.add(pred, gt)) - intersection
    if intersection == 0 or union == 0:
        iou = 0.0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union


def sliding_window_inference(model, image_full, sentences, attentions,
                             category, bert_model, patch_size, img_size,
                             overlap=0.5):
    """
    Sliding window inference on a full-size image.
    Returns aggregated seg_out and exist_out at the original resolution.
    """
    from transforms import sliding_window_positions
    _, _, H, W = image_full.shape

    positions = sliding_window_positions(H, W, patch_size, overlap)
    pred_sum = torch.zeros(1, 2, H, W, device=image_full.device)
    count_map = torch.zeros(1, 1, H, W, device=image_full.device)
    exist_probs = []

    for y0, x0 in positions:
        patch = image_full[:, :, y0:y0+patch_size, x0:x0+patch_size]
        # Resize patch to model input size
        patch_resized = F.interpolate(patch, size=(img_size, img_size),
                                      mode='bilinear', align_corners=False)

        if bert_model is not None:
            last_hidden_states = bert_model(
                sentences, attention_mask=attentions)[0]
            embedding = last_hidden_states.permute(0, 2, 1)
            attn_mask = attentions.unsqueeze(dim=-1)
            seg_patch, exist_out = model(
                patch_resized, embedding, l_mask=attn_mask, category=category)
        else:
            seg_patch, exist_out = model(
                patch_resized, sentences, l_mask=attentions, category=category)

        # Resize prediction back to patch size
        seg_patch_orig = F.interpolate(seg_patch, size=(patch_size, patch_size),
                                       mode='bilinear', align_corners=False)
        pred_sum[:, :, y0:y0+patch_size, x0:x0+patch_size] += seg_patch_orig
        count_map[:, :, y0:y0+patch_size, x0:x0+patch_size] += 1
        exist_probs.append(torch.sigmoid(exist_out).item())

    # Average overlapping regions
    count_map = torch.clamp(count_map, min=1)
    seg_out = pred_sum / count_map
    exist_prob = np.mean(exist_probs)

    return seg_out, exist_prob


def evaluate(model, data_loader, bert_model, patch_size=0, img_size=384):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    acc_ious_ungated = 0
    cum_I_ungated, cum_U_ungated = 0, 0
    mean_IoU_ungated = []
    acc_ious_gated = 0
    cum_I_gated, cum_U_gated = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total   = 0
    neg_total = 0
    neg_correct = 0
    neg_correct_exist = 0
    exist_correct = 0
    exist_total   = 0
    cat_ious = {1: [], 2: [], 3: [], 4: []}

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions, meta = data
            image      = image.cuda(non_blocking=True)
            target     = target.cuda(non_blocking=True)
            sentences  = sentences.cuda(non_blocking=True)
            attentions = attentions.cuda(non_blocking=True)
            sentences  = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            category   = meta['category'].cuda()

            is_pos     = meta['is_pos'].item()
            cat        = meta['category'].item()

            if patch_size > 0:
                # Sliding window inference
                seg_out, exist_prob = sliding_window_inference(
                    model, image, sentences, attentions, category,
                    bert_model, patch_size, img_size)
                # Resize seg_out to match target size
                seg_out = F.interpolate(seg_out, size=target.shape[-2:],
                                        mode='bilinear', align_corners=False)
            else:
                # Standard full-image inference
                if bert_model is not None:
                    last_hidden_states = bert_model(
                        sentences, attention_mask=attentions)[0]
                    embedding  = last_hidden_states.permute(0, 2, 1)
                    attentions = attentions.unsqueeze(dim=-1)
                    seg_out, exist_out = model(
                        image, embedding, l_mask=attentions, category=category)
                else:
                    seg_out, exist_out = model(
                        image, sentences, l_mask=attentions, category=category)
                exist_prob = torch.sigmoid(exist_out).item()

            exist_total += 1
            exist_pred = 1 if exist_prob >= 0.5 else 0
            if exist_pred == is_pos:
                exist_correct += 1

            if is_pos == 1:
                iou_ug, I_ug, U_ug = IoU(seg_out, target)
                acc_ious_ungated += iou_ug
                mean_IoU_ungated.append(iou_ug)
                cum_I_ungated += I_ug
                cum_U_ungated += U_ug
                for n_eval_iou, eval_seg_iou in enumerate(eval_seg_iou_list):
                    seg_correct[n_eval_iou] += (iou_ug >= eval_seg_iou)
                seg_total += 1
                if cat in cat_ious:
                    cat_ious[cat].append(iou_ug)
                if exist_prob >= 0.5:
                    iou_g, I_g, U_g = iou_ug, I_ug, U_ug
                else:
                    iou_g, I_g, U_g = 0.0, 0, 0
                acc_ious_gated += iou_g
                cum_I_gated += I_g
                cum_U_gated += U_g
            else:
                neg_total += 1
                pred = seg_out.cpu().argmax(1)
                if pred.sum() == 0:
                    neg_correct += 1
                if exist_prob < 0.5:
                    neg_correct_exist += 1

    if seg_total > 0:
        mIoU_ug    = np.mean(mean_IoU_ungated)
        overall_ug = float(cum_I_ungated) / float(cum_U_ungated) if cum_U_ungated > 0 else 0.0
        overall_g  = float(cum_I_gated) / float(cum_U_gated) if cum_U_gated > 0 else 0.0
    else:
        mIoU_ug = overall_ug = overall_g = 0.0

    tn_rate       = neg_correct / neg_total if neg_total > 0 else 0.0
    tn_rate_exist = neg_correct_exist / neg_total if neg_total > 0 else 0.0
    exist_acc     = exist_correct / exist_total if exist_total > 0 else 0.0

    print('Final val results:')
    print(f'  Positive samples: {seg_total}')
    print('  [UNGATED] Mean IoU: %.2f%%' % (mIoU_ug * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%.1f = %.2f%%\n' % (
            eval_seg_iou_list[n_eval_iou],
            seg_correct[n_eval_iou] * 100. / max(seg_total, 1))
    results_str += '    [UNGATED] overall IoU = %.2f%%\n' % (overall_ug * 100.)
    results_str += '    [GATED]   overall IoU = %.2f%%\n' % (overall_g * 100.)
    results_str += f'  Negative samples: {neg_total}\n'
    results_str += f'    TN rate (seg head):   {tn_rate*100:.2f}%\n'
    results_str += f'    TN rate (exist head): {tn_rate_exist*100:.2f}%\n'
    results_str += f'  Existence accuracy: {exist_acc*100:.2f}% ({exist_correct}/{exist_total})\n'
    cat_names = {1: 'benign', 2: 'prob_benign', 3: 'prob_suspicious', 4: 'suspicious'}
    results_str += '  Per-category IoU (ungated):\n'
    for cat in [1, 2, 3, 4]:
        if cat_ious[cat]:
            c_iou = np.mean(cat_ious[cat])
            results_str += f'    cls{cat} ({cat_names[cat]}): {c_iou*100:.2f}% (n={len(cat_ious[cat])})\n'
        else:
            results_str += f'    cls{cat} ({cat_names[cat]}): N/A\n'
    print(results_str)
    return 100 * mIoU_ug, 100 * overall_ug


def train_one_epoch(model, criterion, optimizer, data_loader,
                    lr_scheduler, epoch, print_freq, iterations, bert_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header     = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its  = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions, meta = data
        image      = image.cuda(non_blocking=True)
        target     = target.cuda(non_blocking=True)
        sentences  = sentences.cuda(non_blocking=True)
        attentions = attentions.cuda(non_blocking=True)
        sentences  = sentences.squeeze(1)
        attentions = attentions.squeeze(1)
        category   = meta['category'].cuda()

        if bert_model is not None:
            last_hidden_states = bert_model(
                sentences, attention_mask=attentions)[0]
            embedding  = last_hidden_states.permute(0, 2, 1)
            attentions = attentions.unsqueeze(dim=-1)
            seg_out, exist_out = model(
                image, embedding, l_mask=attentions, category=category)
        else:
            seg_out, exist_out = model(
                image, sentences, l_mask=attentions, category=category)

        is_pos = meta['is_pos'].cuda()
        loss = criterion(
            seg_out, exist_out, target,
            is_pos=is_pos, category=category, exist_weight=1.0)

        if epoch >= 5:
            single = model.module if hasattr(model, 'module') else model
            loss = loss + 0.1 * class_embed_contrastive_loss(single)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(),
                             lr=optimizer.param_groups[0]['lr'])


def main(args):
    distributed = args.local_rank >= 0
    if distributed:
        utils.init_distributed_mode(args)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join('./models', args.model_id), exist_ok=True)

    train_transform = get_transform(args)
    if args.patch_size > 0:
        # Sliding window: val images at original resolution (no Resize)
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = T.Compose([
            T.Resize(args.img_size, args.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    dataset,     num_classes = get_dataset('train', train_transform, args)
    dataset_val, _           = get_dataset('val',   val_transform,   args)

    if distributed:
        num_tasks   = utils.get_world_size()
        global_rank = utils.get_rank()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)
    else:
        train_sampler = PatientAwareBatchSampler(
            dataset.annotations, batch_size=args.batch_size,
            drop_last=True, seed=42)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_sampler,
            num_workers=args.workers, pin_memory=args.pin_mem)

    val_sampler = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, sampler=val_sampler,
        num_workers=args.workers)

    print(f'Train: {len(dataset)} slices  |  Val: {len(dataset_val)} slices')

    print(f'Building model: {args.model}')
    model = segmentation.__dict__[args.model](
        pretrained=args.pretrained_swin_weights, args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True)
        single_model = model.module
    else:
        single_model = model

    if args.model != 'lavt_one':
        bert_model = BertModel.from_pretrained(args.ck_bert)
        bert_model.pooler = None
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        if distributed:
            bert_model = torch.nn.parallel.DistributedDataParallel(
                bert_model, device_ids=[args.local_rank])
        single_bert_model = bert_model.module if distributed else bert_model
    else:
        bert_model        = None
        single_bert_model = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'], strict=False)
        if args.model != 'lavt_one' and 'bert_model' in checkpoint:
            single_bert_model.load_state_dict(checkpoint['bert_model'])

    # ── parameters to optimise ────────────────────────────────────────────
    backbone_no_decay, backbone_decay = [], []
    for name, m in single_model.backbone.named_parameters():
        if ('norm' in name or 'absolute_pos_embed' in name
                or 'relative_position_bias_table' in name):
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    class_params = [single_model.class_embed.weight,
                    single_model.class_pos_embed]
    for p in single_model.class_gate.parameters():
        class_params.append(p)

    # exist_head params — directly available, no lazy-build
    exist_params = list(single_model.exist_head.parameters())

    if args.model != 'lavt_one':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {'params': [p for p in single_model.classifier.parameters()
                        if p.requires_grad]},
            {'params': class_params, 'lr': args.lr * 2},
            {'params': exist_params},
            {'params': reduce(operator.concat,
                              [[p for p in single_bert_model.encoder.layer[i].parameters()
                                if p.requires_grad]
                               for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {'params': [p for p in single_model.classifier.parameters()
                        if p.requires_grad]},
            {'params': class_params, 'lr': args.lr * 2},
            {'params': exist_params},
            {'params': reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad]
                               for i in range(10)])},
        ]

    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=args.lr,
        weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    start_time       = time.time()
    iterations       = 0
    best_oIoU        = -0.1
    resume_epoch     = -999
    patience_counter = 0

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']

    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        if hasattr(dataset, 'resample_negatives'):
            dataset.resample_negatives(epoch=epoch)
            if not distributed:
                train_sampler = PatientAwareBatchSampler(
                    dataset.annotations, batch_size=args.batch_size,
                    drop_last=True, seed=42)
                data_loader = torch.utils.data.DataLoader(
                    dataset, batch_sampler=train_sampler,
                    num_workers=args.workers, pin_memory=args.pin_mem)

        if distributed:
            data_loader.sampler.set_epoch(epoch)
        else:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            model, criterion, optimizer, data_loader,
            lr_scheduler, epoch, args.print_freq, iterations, bert_model)

        # Validate every val_every epochs (and always on the last epoch)
        if (epoch + 1) % args.val_every != 0 and epoch != args.epochs - 1:
            print(f'Epoch {epoch}  |  skip validation (every {args.val_every} epochs)')
            continue

        iou, overallIoU = evaluate(
            model, data_loader_val, bert_model,
            patch_size=args.patch_size, img_size=args.img_size)
        print(f'Epoch {epoch}  |  Mean IoU: {iou:.2f}  |  Overall IoU: {overallIoU:.2f}')

        save_ckpt = best_oIoU < overallIoU
        if save_ckpt:
            print(f'  → New best epoch: {epoch}')
            patience_counter = 0
            dict_to_save = {
                'model':        single_model.state_dict(),
                'optimizer':    optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch':        epoch,
                'args':         args,
            }
            if single_bert_model is not None:
                dict_to_save['bert_model'] = single_bert_model.state_dict()
            utils.save_on_master(
                dict_to_save,
                os.path.join(args.output_dir, f'model_best_{args.model_id}.pth'))
            best_oIoU = overallIoU
        else:
            patience_counter += 1
            print(f'  No improvement for {patience_counter}/{args.early_stop} epochs')
            if patience_counter >= args.early_stop:
                print(f'  Early stopping at epoch {epoch}')
                break

    total_time = time.time() - start_time
    print('Training time: {}'.format(
        str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == '__main__':
    parser = get_parser()
    args   = parser.parse_args()
    print(f'Image size: {args.img_size}')
    main(args)