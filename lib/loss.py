import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    """
    Focal + Dice on softmax channel 1 (foreground).

    Per-sample loss split by target.sum():
      - positive: focal + dice
      - negative (all-zero target): focal only, weighted by neg_weight
    Final loss = mean(pos_terms) + neg_weight * mean(neg_terms).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 neg_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.neg_weight = neg_weight
        self.smooth = smooth

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

    def _per_sample_dice(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob_fg = F.softmax(logits, dim=1)[:, 1]      # (B, H, W)
        target_f = target.float()
        inter = (prob_fg * target_f).sum(dim=(1, 2))
        denom = prob_fg.sum(dim=(1, 2)) + target_f.sum(dim=(1, 2))
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice                             # (B,)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        is_pos = target.flatten(1).any(dim=1)         # (B,) bool
        focal = self._per_sample_focal(logits, target)
        dice = self._per_sample_dice(logits, target)

        pos_loss = focal[is_pos] + dice[is_pos]
        neg_loss = focal[~is_pos]

        terms = []
        if pos_loss.numel() > 0:
            terms.append(pos_loss.mean())
        if neg_loss.numel() > 0:
            terms.append(self.neg_weight * neg_loss.mean())
        return sum(terms)
