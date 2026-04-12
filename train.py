"""
train_ln.py
───────────
Training script for LAVT on the LN (Lung Nodule) dataset.

Adapted from train_bc.py — uses LNDataset instead of BCDataset.
4-class lung nodule referring image segmentation (binary output per query).

Example – single GPU:
    python train_ln.py \
        --model lavt \
        --model_id lavt_ln \
        --ln_dataset_root ../dataset \
        --batch-size 8 \
        --lr 0.00005 \
        --wd 1e-2 \
        --swin_type base \
        --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth \
        --window12 \
        --epochs 100 \
        --img_size 384 \
        --workers 4 \
        --pin_mem \
        --output-dir ./checkpoints/ln

Example – multi-GPU (torchrun):
    torchrun --nproc_per_node=2 train_ln.py \
        --model lavt \
        --model_id lavt_ln \
        --ln_dataset_root ../dataset \
        --pretrained_swin_weights /path/to/swin_base.pth \
        --output-dir ./checkpoints/ln \
        --epochs 40 -b 4
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

    # ── dataset ──
    parser.add_argument('--ln_dataset_root', default='../dataset',
                        help='root of the generated LN dataset '
                             '(must contain images/, masks/, annotations/)')

    # ── model ──
    parser.add_argument('--model', default='lavt',
                        help='backbone variant: lavt or lavt_one')
    parser.add_argument('--model_id', default='lavt_ln',
                        help='identifier used for checkpoint naming')
    parser.add_argument('--swin_type', default='base',
                        help='tiny | small | base | large')
    parser.add_argument('--pretrained_swin_weights', default='',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--window12', action='store_true',
                        help='use window-12 Swin (set automatically if '
                             '"window12" appears in pretrained weight filename)')
    parser.add_argument('--mha', default='',
                        help='PWAM head config, e.g. "4-4-4-4"')
    parser.add_argument('--fusion_drop', default=0.0, type=float,
                        help='dropout rate for PWAMs')

    # ── BERT ──
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased')
    parser.add_argument('--ck_bert', default='bert-base-uncased',
                        help='pre-trained BERT weights')

    # ── sampling ──
    parser.add_argument('--neg_ratio', default=2.0, type=float,
                        help='max negatives = neg_ratio × num_positives '
                             '(train only, 0 = keep all)')

    # ── training hyper-params ──
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--img_size', default=384, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', dest='weight_decay')
    parser.add_argument('--amsgrad', action='store_true')
    parser.add_argument('--early_stop', default=10, type=int,
                        help='stop training after N epochs without val IoU improvement')

    # ── I/O ──
    parser.add_argument('--output-dir', default='./checkpoints/ln/')
    parser.add_argument('--resume', default='',
                        help='path to checkpoint to resume from')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='set when loading weights from a DDP checkpoint')

    # ── misc ──
    parser.add_argument('--print-freq', default=10, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--device', default='cuda',
                        help='device for single-GPU mode (ignored in DDP)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='set automatically by torchrun for DDP')

    return parser


# ── patient-aware batch sampler ────────────────────────────────────────────────

class PatientAwareBatchSampler(torch.utils.data.Sampler):
    """
    Yields batches where each sample comes from a different patient.

    Algorithm per epoch:
      1. Shuffle indices within each patient.
      2. Round-robin one index per patient → flat list where consecutive
         indices are from different patients.
      3. Slice the flat list into batches of size B.
         Because of the interleaving, collisions within a batch are rare
         (only possible when num_patients < batch_size).
    """

    def __init__(self, annotations, batch_size, drop_last=True, seed=42):
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.seed       = seed
        self.epoch      = 0

        # group indices by patient
        patient_indices = defaultdict(list)
        for idx, ann in enumerate(annotations):
            patient_indices[ann['patient_id']].append(idx)
        self.patient_indices = dict(patient_indices)
        self.total = sum(len(v) for v in self.patient_indices.values())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        # shuffle within each patient
        queues = {}
        for pid, indices in self.patient_indices.items():
            q = indices[:]
            rng.shuffle(q)
            queues[pid] = q

        # round-robin interleave: one sample per patient per round
        pids = list(queues.keys())
        flat = []
        max_len = max(len(v) for v in queues.values())
        for i in range(max_len):
            rng.shuffle(pids)
            for pid in pids:
                if i < len(queues[pid]):
                    flat.append(queues[pid][i])

        # yield batches
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
        args,
        split=split,
        image_transforms=transform,
        target_transforms=None,
        eval_mode=False,
    )
    num_classes = 2
    return ds, num_classes


# ── transforms ────────────────────────────────────────────────────────────────

def get_transform(args):
    tfms = [
        T.Resize(args.img_size, args.img_size),
        T.RandomHorizontalFlip(flip_prob=0.5),
        T.RandomAffine(
            angle=(-15, 15),
            translate=(0.10, 0.10),
            scale=(0.85, 1.15),
            shear=(-5, 5),
        ),
        T.ToTensor(),
        T.RandomGaussianNoise(std=0.02, p=0.5),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(tfms)

# ── per-category loss weight ───────────────────────────────────────────────────
CLS_LOSS_WEIGHT = {0: 1.0, 1: 2.0, 2: 1.5, 3: 1.0, 4: 1.0}

_WEIGHT_TABLE = torch.tensor(
    [CLS_LOSS_WEIGHT.get(i, 1.0) for i in range(5)],
    dtype=torch.float32
)  # tensor([1.0, 2.0, 1.5, 1.0, 1.0])

# ── per-sample losses ──────────────────────────────────────────────────────────

def dice_loss_per_sample(input, target, smooth=1.0):
    """
    input:  (B, 2, H, W) logits
    target: (B, H, W)    long
    return: (B,)         per-sample dice loss
    """
    prob         = torch.softmax(input, dim=1)[:, 1]          # (B, H, W)
    gt           = target.float()
    intersection = (prob * gt).sum(dim=(1, 2))                 # (B,)
    union        = prob.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))   # (B,)
    dice         = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice                                          # (B,)

def focal_loss_per_sample(input, target, alpha=0.75, gamma=2.0):
    """
    input:  (B, 2, H, W) logits
    target: (B, H, W)    long
    return: (B,)         per-sample focal loss
    """
    ce      = F.cross_entropy(input, target, reduction='none')  # (B, H, W)
    prob    = torch.softmax(input, dim=1)
    p_t     = prob.gather(1, target.unsqueeze(1)).squeeze(1)    # (B, H, W)
    alpha_t = torch.where(
        target == 1,
        torch.tensor(alpha,       device=input.device),
        torch.tensor(1.0 - alpha, device=input.device)
    )
    focal_w = alpha_t * (1.0 - p_t) ** gamma
    return (focal_w * ce).mean(dim=(1, 2))                      # (B,)

# ── main criterion ─────────────────────────────────────────────────────────────

def criterion(seg_out, exist_out, target, is_pos,
              category=None, exist_weight=0.5):
    B = seg_out.shape[0]

    # ── existence loss（全 batch，向量化）─────────────────────────────────
    exist_gt   = is_pos.float().view(B, 1)
    exist_loss = nn.functional.binary_cross_entropy_with_logits(
        exist_out, exist_gt,
        pos_weight=torch.tensor([3.0], device=seg_out.device)
    )

    # ── segmentation loss（只有正樣本）────────────────────────────────────
    has_fg = is_pos.bool()   # (B,)

    if not has_fg.any():
        return exist_weight * exist_loss

    pos_input  = seg_out[has_fg]    # (n_pos, 2, H, W)
    pos_target = target[has_fg]     # (n_pos, H, W)

    # per-sample loss
    bce_per   = nn.functional.cross_entropy(
        pos_input, pos_target, reduction='none'
    ).mean(dim=(1, 2))                                    # (n_pos,)
    dice_per  = dice_loss_per_sample(pos_input, pos_target)   # (n_pos,)
    focal_per = focal_loss_per_sample(pos_input, pos_target)  # (n_pos,)

    loss_per = bce_per + dice_per + focal_per             # (n_pos,)

    # cls weight（向量化，無 for loop）
    if category is not None:
        pos_cat = category[has_fg]                        # (n_pos,)
        cls_w   = _WEIGHT_TABLE.to(seg_out.device)[pos_cat]  # (n_pos,)
        seg_loss = (loss_per * cls_w).sum() / cls_w.sum()
    else:
        seg_loss = loss_per.mean()

    return seg_loss + exist_weight * exist_loss

def class_embed_contrastive_loss(model):
    """
    Push the 4 category embeddings apart (computed once per iteration).
    Uses all-pairs cosine similarity with margin.
    """
    # get class embeddings for cls 1-4 (skip index 0 = normal)
    embs = model.class_embed.weight[1:5]  # (4, 768)
    embs = nn.functional.normalize(embs, dim=-1)
    sim = torch.mm(embs, embs.t())  # (4, 4)
    # exclude diagonal
    mask = ~torch.eye(4, dtype=torch.bool, device=sim.device)
    # push all off-diagonal cosine similarities toward 0
    return sim[mask].pow(2).mean()

# ── IoU ───────────────────────────────────────────────────────────────────────

def IoU(pred, gt):
    pred = pred.argmax(1)
    intersection = torch.sum(torch.mul(pred, gt))
    union        = torch.sum(torch.add(pred, gt)) - intersection
    if intersection == 0 or union == 0:
        iou = 0.0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, data_loader, bert_model):
    """
    Evaluate on validation set.
    Metrics are computed ONLY on positive samples (is_pos=1) where
    there is a real ground-truth mask. Negative samples are evaluated
    separately via true-negative rate (model predicts no foreground).
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    total_its = 0

    # positive sample metrics
    acc_ious  = 0
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total   = 0
    mean_IoU    = []

    # negative sample metrics
    neg_total   = 0
    neg_correct = 0   # TN via seg head (pred all bg)
    neg_correct_exist = 0  # TN via existence head (exist_prob < 0.5)

    # existence head accuracy
    exist_correct = 0
    exist_total   = 0

    # per-category positive metrics
    cat_ious = {1: [], 2: [], 3: [], 4: []}

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions, meta = data
            image     = image.cuda(non_blocking=True)
            target    = target.cuda(non_blocking=True)
            sentences = sentences.cuda(non_blocking=True)
            attentions = attentions.cuda(non_blocking=True)

            sentences  = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            category = meta['category'].cuda()

            if bert_model is not None:
                last_hidden_states = bert_model(
                    sentences, attention_mask=attentions
                )[0]
                embedding  = last_hidden_states.permute(0, 2, 1)
                attentions = attentions.unsqueeze(dim=-1)
                seg_out, exist_out = model(image, embedding, l_mask=attentions, category=category)
            else:
                seg_out, exist_out = model(image, sentences, l_mask=attentions, category=category)

            is_pos = meta['is_pos'].item()
            cat    = meta['category'].item()
            exist_prob = torch.sigmoid(exist_out).item()

            # existence accuracy
            exist_total += 1
            exist_pred = 1 if exist_prob >= 0.5 else 0
            if exist_pred == is_pos:
                exist_correct += 1

            if is_pos == 1:
                # gate segmentation by existence head
                if exist_prob < 0.5:
                    iou, I, U = 0.0, 0, 0
                else:
                    iou, I, U = IoU(seg_out, target)
                acc_ious += iou
                mean_IoU.append(iou)
                cum_I += I
                cum_U += U
                for n_eval_iou, eval_seg_iou in enumerate(eval_seg_iou_list):
                    seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
                seg_total += 1
                if cat in cat_ious:
                    cat_ious[cat].append(iou)
            else:
                neg_total += 1
                # TN via seg head
                pred = seg_out.cpu().argmax(1)
                if pred.sum() == 0:
                    neg_correct += 1
                # TN via existence head
                if exist_prob < 0.5:
                    neg_correct_exist += 1

    if seg_total > 0:
        mean_IoU = np.array(mean_IoU)
        mIoU     = np.mean(mean_IoU)
        iou      = acc_ious / seg_total
        overall  = float(cum_I) / float(cum_U) if cum_U > 0 else 0.0
    else:
        mIoU = iou = overall = 0.0

    tn_rate = neg_correct / neg_total if neg_total > 0 else 0.0
    tn_rate_exist = neg_correct_exist / neg_total if neg_total > 0 else 0.0
    exist_acc = exist_correct / exist_total if exist_total > 0 else 0.0

    print('Final val results:')
    print(f'  Positive samples: {seg_total}')
    print('  Mean IoU: %.2f%%' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%.1f = %.2f%%\n' % (
            eval_seg_iou_list[n_eval_iou],
            seg_correct[n_eval_iou] * 100. / max(seg_total, 1),
        )
    results_str += '    overall IoU = %.2f%%\n' % (overall * 100.)
    results_str += f'  Negative samples: {neg_total}\n'
    results_str += f'    TN rate (seg head):   {tn_rate*100:.2f}%\n'
    results_str += f'    TN rate (exist head): {tn_rate_exist*100:.2f}%\n'
    results_str += f'  Existence accuracy: {exist_acc*100:.2f}% ({exist_correct}/{exist_total})\n'
    cat_names = {1: 'benign', 2: 'prob_benign', 3: 'prob_suspicious', 4: 'suspicious'}
    results_str += '  Per-category IoU:\n'
    for cat in [1, 2, 3, 4]:
        if cat_ious[cat]:
            c_iou = np.mean(cat_ious[cat])
            results_str += f'    cls{cat} ({cat_names[cat]}): {c_iou*100:.2f}% (n={len(cat_ious[cat])})\n'
        else:
            results_str += f'    cls{cat} ({cat_names[cat]}): N/A\n'
    print(results_str)

    return 100 * iou, 100 * overall


# ── train one epoch ───────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, optimizer, data_loader,
                    lr_scheduler, epoch, print_freq, iterations, bert_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header      = 'Epoch: [{}]'.format(epoch)
    train_loss  = 0
    total_its   = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions, meta = data
        image     = image.cuda(non_blocking=True)
        target    = target.cuda(non_blocking=True)
        sentences = sentences.cuda(non_blocking=True)
        attentions = attentions.cuda(non_blocking=True)

        sentences  = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        category = meta['category'].cuda()

        if bert_model is not None:
            last_hidden_states = bert_model(
                sentences, attention_mask=attentions
            )[0]
            embedding  = last_hidden_states.permute(0, 2, 1)
            attentions = attentions.unsqueeze(dim=-1)
            seg_out, exist_out = model(image, embedding, l_mask=attentions, category=category)
        else:
            seg_out, exist_out = model(image, sentences, l_mask=attentions, category=category)

        is_pos = meta['is_pos'].cuda()
        loss = criterion(seg_out, 
                         exist_out, 
                         target,
                         is_pos=is_pos, 
                         category=category, 
                         exist_weight = 0.5
        )
        
        # contrastive loss on class embeddings (once per iteration)
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


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    # ── distributed / single-GPU setup ───────────────────────────────────
    distributed = args.local_rank >= 0
    if distributed:
        utils.init_distributed_mode(args)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join('./models', args.model_id), exist_ok=True)

    # ── datasets ─────────────────────────────────────────────────────────
    train_transform = get_transform(args)
    val_transform = T.Compose([
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    dataset,      num_classes = get_dataset('train', train_transform, args)
    dataset_val,  _           = get_dataset('val',   val_transform,   args)

    if distributed:
        num_tasks    = utils.get_world_size()
        global_rank  = utils.get_rank()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print(f'local_rank {args.local_rank} / global_rank {global_rank} '
              f'built train dataset.')
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    else:
        train_sampler = PatientAwareBatchSampler(
            dataset.annotations, batch_size=args.batch_size,
            drop_last=True, seed=42,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
        )

    val_sampler = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        sampler=val_sampler,
        num_workers=args.workers,
    )

    print(f'Train: {len(dataset)} slices  |  Val: {len(dataset_val)} slices')

    # ── model ─────────────────────────────────────────────────────────────
    print(f'Building model: {args.model}')
    model = segmentation.__dict__[args.model](
        pretrained=args.pretrained_swin_weights, args=args
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )
        single_model = model.module
    else:
        single_model = model

    # ── BERT encoder ──────────────────────────────────────────────────────
    if args.model != 'lavt_one':
        bert_model = BertModel.from_pretrained(args.ck_bert)
        bert_model.pooler = None
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        if distributed:
            bert_model = torch.nn.parallel.DistributedDataParallel(
                bert_model, device_ids=[args.local_rank]
            )
        single_bert_model = bert_model.module if distributed else bert_model
    else:
        bert_model        = None
        single_bert_model = None

    # ── resume ────────────────────────────────────────────────────────────
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        if args.model != 'lavt_one' and 'bert_model' in checkpoint:
            single_bert_model.load_state_dict(checkpoint['bert_model'])

    # ── parameters to optimise ────────────────────────────────────────────
    is_unet = args.model.startswith('unet_')

    if is_unet:
        # UNet: all model params in one group (backbone, pwam, decoder, heads)
        no_decay, decay = [], []
        for name, p in single_model.named_parameters():
            if not p.requires_grad:
                continue
            if 'norm' in name or 'bias' in name:
                no_decay.append(p)
            else:
                decay.append(p)
        params_to_optimize = [
            {'params': no_decay, 'weight_decay': 0.0},
            {'params': decay},
        ]
        if bert_model is not None:
            params_to_optimize.append(
                {'params': reduce(operator.concat,
                                  [[p for p in single_bert_model.encoder.layer[i].parameters()
                                    if p.requires_grad]
                                   for i in range(10)])},
            )
    else:
        # LAVT / LAVT One: original param grouping + exist_head
        backbone_no_decay, backbone_decay = [], []
        for name, m in single_model.backbone.named_parameters():
            if ('norm' in name or 'absolute_pos_embed' in name
                    or 'relative_position_bias_table' in name):
                backbone_no_decay.append(m)
            else:
                backbone_decay.append(m)

        if args.model != 'lavt_one':
            params_to_optimize = [
                {'params': backbone_no_decay, 'weight_decay': 0.0},
                {'params': backbone_decay},
                {'params': [p for p in single_model.classifier.parameters()
                            if p.requires_grad]},
                {'params': [p for p in single_model.exist_head.parameters()
                            if p.requires_grad]},
                {'params': [single_model.class_embed.weight,
                            single_model.class_pos_embed]},
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
                {'params': [p for p in single_model.exist_head.parameters()
                            if p.requires_grad]},
                {'params': [single_model.class_embed.weight,
                            single_model.class_pos_embed]},
                {'params': reduce(operator.concat,
                                  [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                    if p.requires_grad]
                                   for i in range(10)])},
            ]

    # ── optimizer & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9,
    )

    # ── housekeeping ──────────────────────────────────────────────────────
    start_time      = time.time()
    iterations      = 0
    best_oIoU       = -0.1
    resume_epoch    = -999
    patience_counter = 0

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        # resample negatives each epoch (different subset, same ratio)
        if hasattr(dataset, 'resample_negatives'):
            dataset.resample_negatives(epoch=epoch)
            # rebuild batch sampler with new annotations
            if not distributed:
                train_sampler = PatientAwareBatchSampler(
                    dataset.annotations, batch_size=args.batch_size,
                    drop_last=True, seed=42,
                )
                data_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_sampler=train_sampler,
                    num_workers=args.workers,
                    pin_memory=args.pin_mem,
                )

        if distributed:
            data_loader.sampler.set_epoch(epoch)
        else:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            model, criterion, optimizer, data_loader,
            lr_scheduler, epoch, args.print_freq, iterations, bert_model,
        )

        iou, overallIoU = evaluate(model, data_loader_val, bert_model)
        print(f'Epoch {epoch}  |  Mean IoU: {iou:.2f}  |  Overall IoU: {overallIoU:.2f}')

        save_checkpoint = best_oIoU < overallIoU
        if save_checkpoint:
            print(f'  → New best epoch: {epoch}')
            patience_counter = 0
            if single_bert_model is not None:
                dict_to_save = {
                    'model':        single_model.state_dict(),
                    'bert_model':   single_bert_model.state_dict(),
                    'optimizer':    optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch':        epoch,
                    'args':         args,
                }
            else:
                dict_to_save = {
                    'model':        single_model.state_dict(),
                    'optimizer':    optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch':        epoch,
                    'args':         args,
                }
            utils.save_on_master(
                dict_to_save,
                os.path.join(args.output_dir,
                             f'model_best_{args.model_id}.pth'),
            )
            best_oIoU = overallIoU
        else:
            patience_counter += 1
            print(f'  No improvement for {patience_counter}/{args.early_stop} epochs')
            if patience_counter >= args.early_stop:
                print(f'  Early stopping at epoch {epoch}')
                break

    total_time = time.time() - start_time
    print('Training time: {}'.format(
        str(datetime.timedelta(seconds=int(total_time)))
    ))


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = get_parser()
    args   = parser.parse_args()
    print(f'Image size: {args.img_size}')
    main(args)
