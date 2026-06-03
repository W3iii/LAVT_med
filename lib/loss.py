import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    """
    Focal + Dice on softmax channel 1 (foreground).

    All samples (positive and negative) are treated equally — same as
    nnU-Net's CE+Dice approach.  Focal loss covers all pixels; Dice is
    computed only over positive samples (those with at least one fg pixel).

    batch_dice=True (nnUNet-style): Dice accumulated across all positive
    samples as one pool before dividing — prevents tiny nodules from being
    dominated by the smoothing term.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 smooth: float = 1.0, batch_dice: bool = True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth
        self.batch_dice = batch_dice

    def _per_sample_focal(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        target_oh = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
        pt     = (prob     * target_oh).sum(dim=1)
        log_pt = (log_prob * target_oh).sum(dim=1)
        alpha_t = target_oh[:, 1] * self.alpha + target_oh[:, 0] * (1.0 - self.alpha)
        focal = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt
        return focal.mean(dim=(1, 2))                 # (B,)

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor,
                   is_pos: torch.Tensor) -> torch.Tensor:
        if not is_pos.any():
            return logits.sum() * 0.0
        prob_fg = F.softmax(logits, dim=1)[:, 1]
        if self.batch_dice:
            prob_pos = prob_fg[is_pos]
            tgt_pos  = target[is_pos].float()
            inter = (prob_pos * tgt_pos).sum()
            denom = prob_pos.sum() + tgt_pos.sum()
            return 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)
        else:
            target_f = target.float()
            inter = (prob_fg * target_f).sum(dim=(1, 2))
            denom = prob_fg.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
            dice  = (2.0 * inter + self.smooth) / (denom + self.smooth)
            return (1.0 - dice[is_pos]).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        is_pos = target.flatten(1).any(dim=1)
        return self._per_sample_focal(logits, target).mean() + \
               self._dice_loss(logits, target, is_pos)


class DeepSupervisionLoss(nn.Module):
    """
    Wraps a per-scale criterion for nnU-Net-style deep supervision.

    During training the model returns a list of logits
    [full_res, half_res, quarter_res] (coarsest first in the list means
    the first element is already upsampled to the target; subsequent
    elements are at 1/2 and 1/4 of target resolution).

    weights default to nnU-Net's (1, 0.5, 0.25) — unnormalised is fine
    because they just scale gradient magnitude, not the loss value itself.
    """

    def __init__(self, criterion, weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.criterion = criterion
        self.weights   = weights

    def forward(self, outputs, target):
        if not isinstance(outputs, list):
            return self.criterion(outputs, target)

        total = outputs[0].new_zeros(())
        for w, logit in zip(self.weights, outputs):
            h, w_sz = logit.shape[-2:]
            if (h, w_sz) != target.shape[-2:]:
                # max-pool preserves any foreground pixel in the window
                t = F.adaptive_max_pool2d(
                    target.unsqueeze(1).float(), (h, w_sz)
                ).squeeze(1).long()
            else:
                t = target
            total = total + w * self.criterion(logit, t)
        return total
