import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    """
    Focal + Dice on softmax channel 1 (foreground).

    batch_dice=True  (nnUNet-style, default):
      Dice is computed over ALL positive samples in the batch as one pool.
      Inter and denom are accumulated across samples before dividing, so
      tiny nodules contribute proportionally instead of being drowned by
      the smooth term in per-sample Dice.

    batch_dice=False (original behaviour):
      Dice computed per sample, then averaged — can collapse to near-zero
      gradient for small nodules.

    Per-sample focal loss is always computed (positive + negative).
    Final loss = (focal_pos + dice_pos).mean() + neg_weight * focal_neg.mean()
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 neg_weight: float = 0.5, smooth: float = 1.0,
                 batch_dice: bool = True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.neg_weight = neg_weight
        self.smooth = smooth
        self.batch_dice = batch_dice

    def _per_sample_focal(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, 2, H, W); target: (B, H, W) in {0, 1}
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()

        target_oh = F.one_hot(target, num_classes=2).permute(0, 3, 1, 2).float()
        pt = (prob * target_oh).sum(dim=1)            # (B, H, W)
        log_pt = (log_prob * target_oh).sum(dim=1)    # (B, H, W)

        alpha_t = target_oh[:, 1] * self.alpha + target_oh[:, 0] * (1.0 - self.alpha)
        focal = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt
        return focal.mean(dim=(1, 2))                 # (B,)

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor,
                   is_pos: torch.Tensor) -> torch.Tensor:
        """Returns scalar Dice loss over positive samples."""
        if not is_pos.any():
            return logits.sum() * 0.0                 # zero with grad

        prob_fg = F.softmax(logits, dim=1)[:, 1]     # (B, H, W)

        if self.batch_dice:
            # Accumulate inter/denom across all positive samples before dividing.
            prob_pos = prob_fg[is_pos]                # (n_pos, H, W)
            tgt_pos = target[is_pos].float()
            inter = (prob_pos * tgt_pos).sum()
            denom = prob_pos.sum() + tgt_pos.sum()
            return 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)
        else:
            target_f = target.float()
            inter = (prob_fg * target_f).sum(dim=(1, 2))
            denom = prob_fg.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
            dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
            return (1.0 - dice[is_pos]).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        is_pos = target.flatten(1).any(dim=1)         # (B,) bool
        focal = self._per_sample_focal(logits, target)

        terms = []
        if is_pos.any():
            terms.append(focal[is_pos].mean() + self._dice_loss(logits, target, is_pos))
        if (~is_pos).any():
            terms.append(self.neg_weight * focal[~is_pos].mean())
        return sum(terms)
