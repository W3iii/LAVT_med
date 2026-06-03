import datetime
import os
import time
from functools import reduce
import operator

import numpy as np
import torch
import torch.utils.data
from scipy.ndimage import label as cc_label

from lib import segmentation
from lib.loss import FocalDiceLoss, DeepSupervisionLoss
from data.dataset_lung_nodule import LungNoduleDataset
from data.sampler import PatientAwareBatchSampler

import transforms as T
import utils


def get_transform(args, is_train=False):
    h = args.img_h if args.img_size is None else args.img_size
    w = args.img_w if args.img_size is None else args.img_size
    transforms = [T.Resize(h, w)]
    if is_train:
        transforms += [
            T.RandomFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=30, p=0.5),
            T.RandomScaleAndCrop(scale_range=(0.85, 1.25), p=0.5),
        ]
    transforms.append(T.ToTensor())
    if is_train:
        transforms += [
            T.GaussianNoise(std_range=(0.0, 0.1), p=0.15),
            T.GammaAugmentation(gamma_range=(0.7, 1.5), p=0.3),
        ]
    if getattr(args, 'backbone', 'swin') != 'nnunet':
        transforms.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    return T.Compose(transforms)


def get_dataset(split, transform, args):
    return LungNoduleDataset(
        data_root=args.data_root,
        split=split,
        transforms=transform,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )


_CC_THRS = (0.3, 0.5)


def _cc_pr_counts(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """
    Per-slice CC-level matching.  For each GT CC and each pred CC, find the
    best Dice against any overlapping CC on the other side.

    Returns:
        best_gt   (n_gt,)   – best Dice each GT CC achieved against pred
        best_pred (n_pred,) – best Dice each pred CC achieved against GT
    """
    pred_cc, n_pred = cc_label(pred_mask)
    gt_cc,   n_gt   = cc_label(gt_mask)

    best_gt   = np.zeros(n_gt,   dtype=np.float32)
    best_pred = np.zeros(n_pred, dtype=np.float32)

    for g in range(1, n_gt + 1):
        gt_g    = gt_cc == g
        gt_sum  = int(gt_g.sum())
        for p in np.unique(pred_cc[gt_g]):
            if p == 0:
                continue
            pred_p = pred_cc == p
            inter  = int((pred_p & gt_g).sum())
            d      = 2 * inter / (int(pred_p.sum()) + gt_sum)
            if d > best_gt[g - 1]:
                best_gt[g - 1] = d
            if d > best_pred[p - 1]:
                best_pred[p - 1] = d

    return best_gt, best_pred


@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_chunks = []
    cum_inter = torch.zeros((), dtype=torch.float64, device='cuda')
    cum_union = torch.zeros((), dtype=torch.float64, device='cuda')
    n_pos = 0
    n_neg = 0
    n_tn  = 0
    # CC nodule-level counters: gt_hits, n_gt, pred_hits, n_pred
    cc_counts = {thr: [0, 0, 0, 0] for thr in _CC_THRS}

    for image, target in metric_logger.log_every(data_loader, 100, header):
        image  = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)        # (B, H, W) {0, 1}

        logits = model(image)                          # (B, 2, H, W)
        pred   = logits.argmax(dim=1)                  # (B, H, W) {0, 1}

        target_flat = target.flatten(1)
        pred_flat   = pred.flatten(1)
        is_pos = target_flat.any(dim=1)                # (B,) bool

        inter = (pred_flat * target_flat).sum(dim=1).double()
        union = (pred_flat.sum(dim=1) + target_flat.sum(dim=1)).double() - inter
        iou   = torch.where(union > 0, inter / union, torch.zeros_like(inter))

        if is_pos.any():
            iou_chunks.append(iou[is_pos].cpu())
            cum_inter += inter[is_pos].sum()
            cum_union += union[is_pos].sum()
            n_pos += int(is_pos.sum().item())

            pred_np = pred[is_pos].cpu().numpy().astype(bool)
            gt_np   = target[is_pos].cpu().numpy().astype(bool)
            for ps, gs in zip(pred_np, gt_np):
                best_gt, best_pred = _cc_pr_counts(ps, gs)
                for thr in _CC_THRS:
                    c = cc_counts[thr]
                    c[0] += int((best_gt   >= thr).sum())
                    c[1] += len(best_gt)
                    c[2] += int((best_pred >= thr).sum())
                    c[3] += len(best_pred)

        neg_mask = ~is_pos
        if neg_mask.any():
            pred_neg_sum = pred_flat[neg_mask].sum(dim=1)
            n_tn += int((pred_neg_sum == 0).sum().item())
            n_neg += int(neg_mask.sum().item())

    mean_iou    = (torch.cat(iou_chunks).mean().item() if iou_chunks else 0.0) * 100.0
    overall_iou = (cum_inter / cum_union).item() * 100.0 if cum_union.item() > 0 else 0.0
    tn_rate     = (n_tn / n_neg) * 100.0 if n_neg > 0 else 0.0

    print('Final results:')
    print(f'  Mean IoU:    {mean_iou:.2f}  ({n_pos} positive samples)')
    print(f'  Overall IoU: {overall_iou:.2f}')
    print(f'  TN rate:     {tn_rate:.2f}  ({n_tn}/{n_neg} negatives all-zero)')
    for thr in _CC_THRS:
        gh, ng, ph, np_ = cc_counts[thr]
        rec  = gh / ng   * 100.0 if ng   > 0 else 0.0
        prec = ph / np_  * 100.0 if np_  > 0 else 0.0
        print(f'  Nodule Dice>{thr:.1f}: P={prec:.1f}% ({ph}/{np_})  R={rec:.1f}% ({gh}/{ng})')

    return mean_iou, overall_iou, tn_rate


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler,
                    epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = f'Epoch: [{epoch}]'

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logits = model(image)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def _build_optimizer(single_model, args):
    backbone_no_decay, backbone_decay = [], []
    for name, p in single_model.backbone.named_parameters():
        if not p.requires_grad:
            continue
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(p)
        else:
            backbone_decay.append(p)

    bert_params = reduce(
        operator.concat,
        [[p for p in single_model.text_encoder.encoder.layer[i].parameters() if p.requires_grad]
         for i in range(10)],
    )

    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0},
        {'params': backbone_decay},
        {'params': [p for p in single_model.classifier.parameters() if p.requires_grad]},
        {'params': [single_model.soft_tokens]},
        {'params': bert_params},
    ]
    return torch.optim.AdamW(params_to_optimize, lr=args.lr,
                             weight_decay=args.weight_decay, amsgrad=args.amsgrad)


def main(args):
    distributed = args.local_rank is not None and args.local_rank >= 0
    if distributed:
        utils.init_distributed_mode(args)
    else:
        torch.cuda.set_device(0)
        if args.output_dir:
            utils.mkdir(args.output_dir)
        if args.model_id:
            utils.mkdir(os.path.join('./models/', args.model_id))

    print(f'Image size: {args.img_size}')
    print(f'Distributed: {distributed}')

    dataset_train = get_dataset('train', get_transform(args, is_train=True), args)
    dataset_val   = get_dataset('val',   get_transform(args, is_train=False), args)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train, num_replicas=utils.get_world_size(),
            rank=utils.get_rank(), shuffle=True)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)
    else:
        train_sampler = PatientAwareBatchSampler(
            dataset_train, batch_size=args.batch_size,
            drop_last=True, shuffle=True, seed=args.seed,
            fg_fraction=args.fg_fraction)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_sampler=train_sampler,
            num_workers=args.workers, pin_memory=args.pin_mem)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.pin_mem)

    print(f'Built train dataset: {len(dataset_train)} samples '
          f'({len(dataset_train.positives)} pos + {len(dataset_train.samples) - len(dataset_train.positives)} neg)')
    print(f'Built val dataset:   {len(dataset_val)} samples')

    print(f'Model: {args.model}')
    model = segmentation.__dict__[args.model](
        pretrained=args.pretrained_swin_weights, args=args)
    model.cuda()

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True)
        single_model = model.module
    else:
        single_model = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])

    optimizer = _build_optimizer(single_model, args)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader_train) * args.epochs)) ** 0.9,
    )

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -1

    criterion = DeepSupervisionLoss(
        FocalDiceLoss(gamma=2.0, alpha=0.9, batch_dice=args.batch_dice)
    ).cuda()

    start_time = time.time()
    best_overall_iou = -1.0

    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        dataset_train.resample_negatives(epoch)
        train_sampler.set_epoch(epoch)

        train_one_epoch(model, criterion, optimizer, data_loader_train,
                        lr_scheduler, epoch, args.print_freq)

        mean_iou, overall_iou, tn_rate = evaluate(model, data_loader_val)
        print(f'Epoch {epoch}: Mean IoU={mean_iou:.2f}, Overall IoU={overall_iou:.2f}, '
              f'TN rate={tn_rate:.2f}')

        if overall_iou > best_overall_iou:
            print(f'Better epoch: {epoch}')
            utils.save_on_master(
                {
                    'model': single_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                },
                os.path.join(args.output_dir, f'model_best_{args.model_id}.pth'),
            )
            best_overall_iou = overall_iou

    total_time = time.time() - start_time
    print(f'Training time {datetime.timedelta(seconds=int(total_time))}')


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)
