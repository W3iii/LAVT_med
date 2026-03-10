"""
train_bc.py
───────────
Training script for LAVT on the BC (Breast Cancer) dataset.

Adapted from the original train.py (LAVT-RIS).
Key differences:
  - Uses BCDataset (data/dataset_bc.py) instead of ReferDataset
  - Adds --bc_dataset_root argument
  - Supports both single-GPU and multi-GPU (DDP) training
  - No dependency on the REFER API / refcoco splits

Example – single GPU:
    python train_bc.py `
    --model lavt `
    --model_id lavt_bc `
    --bc_dataset_root ../dataset `
    --batch-size 8 `
    --lr 0.00005 `
    --wd 1e-2 `
    --swin_type base `
    --pretrained_swin_weights ./pretrained_weights/swin_base_patch4_window12_384_22k.pth `
    --window12 `
    --epochs 100 `
    --img_size 384 `
    --workers 4 `
    --pin_mem `
    --output-dir ./checkpoints/bc `
    2>&1 | Tee-Object -FilePath ./models/lavt_bc/output.log

Example – multi-GPU (torchrun):
    torchrun --nproc_per_node=2 train_bc.py \
        --model lavt \
        --model_id lavt_bc \
        --bc_dataset_root ../dataset \
        --pretrained_swin_weights /path/to/swin_base.pth \
        --output-dir ./checkpoints/bc \
        --epochs 40 -b 4
"""

import argparse
import datetime
import gc
import os
import time
from functools import reduce
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch import nn

import torchvision

import transforms as T
import utils
from transformers import BertModel
from lib import segmentation


# ── argument parsing ──────────────────────────────────────────────────────────

def get_parser():
    parser = argparse.ArgumentParser(description='LAVT-BC training')

    # ── dataset ──
    parser.add_argument('--bc_dataset_root', default='../dataset',
                        help='root of the generated BC dataset '
                             '(must contain images/, masks/, annotations/)')

    # ── model ──
    parser.add_argument('--model', default='lavt',
                        help='backbone variant: lavt or lavt_one')
    parser.add_argument('--model_id', default='lavt_bc',
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

    # ── training hyper-params ──
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--img_size', default=384, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', dest='weight_decay')
    parser.add_argument('--amsgrad', action='store_true')

    # ── I/O ──
    parser.add_argument('--output-dir', default='./checkpoints/bc/')
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


# ── dataset factory ───────────────────────────────────────────────────────────

def get_dataset(split, transform, args):
    from data.dataset_bc import BCDataset
    # eval_mode=False during training: pick one random sentence per sample
    # (same as original LAVT train.py). eval_mode=True is only for final test
    # inference where all sentences are evaluated; it returns 3-D tensors that
    # BERT cannot directly consume without an extra loop.
    ds = BCDataset(
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
    # ImageNet mean/std works fine for 3-channel grayscale replicated images
    tfms = [
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(tfms)


# ── loss ──────────────────────────────────────────────────────────────────────

def dice_loss(input, target, smooth=1.0):
    """Soft Dice Loss over the tumour class (class-1 probability)."""
    prob = torch.softmax(input, dim=1)[:, 1]          # (B, H, W)  tumour prob
    gt   = target.float()                              # (B, H, W)
    intersection = (prob * gt).sum(dim=(1, 2))
    dice = (2.0 * intersection + smooth) / (prob.sum(dim=(1, 2)) + gt.sum(dim=(1, 2)) + smooth)
    return 1.0 - dice.mean()


def criterion(input, target):
    """Dice + BCE combination loss – better suited for imbalanced medical segmentation."""
    bce  = nn.functional.cross_entropy(input, target)
    dice = dice_loss(input, target)
    return bce + dice


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
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    total_its = 0
    acc_ious  = 0
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total   = 0
    mean_IoU    = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions = data
            image     = image.cuda(non_blocking=True)
            target    = target.cuda(non_blocking=True)
            sentences = sentences.cuda(non_blocking=True)
            attentions = attentions.cuda(non_blocking=True)

            sentences  = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            if bert_model is not None:
                last_hidden_states = bert_model(
                    sentences, attention_mask=attentions
                )[0]
                embedding  = last_hidden_states.permute(0, 2, 1)   # (B, 768, N_l)
                attentions = attentions.unsqueeze(dim=-1)           # (B, N_l, 1)
                output     = model(image, embedding, l_mask=attentions)
            else:
                output = model(image, sentences, l_mask=attentions)

            iou, I, U = IoU(output, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou, eval_seg_iou in enumerate(eval_seg_iou_list):
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
            seg_total += 1

    mean_IoU = np.array(mean_IoU)
    mIoU     = np.mean(mean_IoU)
    iou      = acc_ious / total_its

    print('Final val results:')
    print('  Mean IoU: %.2f%%' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%.1f = %.2f%%\n' % (
            eval_seg_iou_list[n_eval_iou],
            seg_correct[n_eval_iou] * 100. / seg_total,
        )
    results_str += '    overall IoU = %.2f%%\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * iou, 100 * cum_I / cum_U


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
        image, target, sentences, attentions = data
        image     = image.cuda(non_blocking=True)
        target    = target.cuda(non_blocking=True)
        sentences = sentences.cuda(non_blocking=True)
        attentions = attentions.cuda(non_blocking=True)

        sentences  = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(
                sentences, attention_mask=attentions
            )[0]
            embedding  = last_hidden_states.permute(0, 2, 1)   # (B, 768, N_l)
            attentions = attentions.unsqueeze(dim=-1)           # (B, N_l, 1)
            output     = model(image, embedding, l_mask=attentions)
        else:
            output = model(image, sentences, l_mask=attentions)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(),
                             lr=optimizer.param_groups[0]['lr'])

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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
    transform = get_transform(args)
    dataset,      num_classes = get_dataset('train', transform, args)
    dataset_val,  _           = get_dataset('val',   transform, args)

    if distributed:
        num_tasks    = utils.get_world_size()
        global_rank  = utils.get_rank()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print(f'local_rank {args.local_rank} / global_rank {global_rank} '
              f'built train dataset.')
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)

    val_sampler = torch.utils.data.SequentialSampler(dataset_val)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
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
        bert_model.pooler = None   # work-around for Transformers 3.0.2 bug
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
    start_time   = time.time()
    iterations   = 0
    best_oIoU    = -0.1
    resume_epoch = -999

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        if distributed:
            data_loader.sampler.set_epoch(epoch)

        train_one_epoch(
            model, criterion, optimizer, data_loader,
            lr_scheduler, epoch, args.print_freq, iterations, bert_model,
        )

        iou, overallIoU = evaluate(model, data_loader_val, bert_model)
        print(f'Epoch {epoch}  |  Mean IoU: {iou:.2f}  |  Overall IoU: {overallIoU:.2f}')

        save_checkpoint = best_oIoU < overallIoU
        if save_checkpoint:
            print(f'  → New best epoch: {epoch}')
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
