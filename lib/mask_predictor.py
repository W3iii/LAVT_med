import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


# ---------------------------------------------------------------------------
# nnUNet-style decoder
# ---------------------------------------------------------------------------

class _ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class NnUNetDecoding(nn.Module):
    """
    nnUNet-style top-down decoder (InstanceNorm + LeakyReLU).

    Each decode stage: bilinear upsample → concat skip → 2× ConvNormAct.

    Args:
        c4, c3, c2, c1: encoder output channels (coarse→fine).
        n_conv_per_stage: conv blocks per decode stage (default 2).
        deep_supervision: if True, return [logit_c1, logit_c2, logit_c3]
            during training (weights 1, 0.5, 0.25 applied by the loss).
            At eval time always returns the single full-res logit.
    """

    def __init__(self, c4, c3, c2, c1, n_conv_per_stage=2,
                 deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        def _stage(in_ch, out_ch):
            layers = [_ConvNormAct(in_ch, out_ch)]
            for _ in range(n_conv_per_stage - 1):
                layers.append(_ConvNormAct(out_ch, out_ch))
            return nn.Sequential(*layers)

        self.dec43 = _stage(c4 + c3, c3)
        self.dec32 = _stage(c3 + c2, c2)
        self.dec21 = _stage(c2 + c1, c1)
        self.seg_head = nn.Conv2d(c1, 2, 1)

        if deep_supervision:
            self.seg_head_ds2 = nn.Conv2d(c2, 2, 1)   # after dec32 (½ res)
            self.seg_head_ds3 = nn.Conv2d(c3, 2, 1)   # after dec43 (¼ res)

    @staticmethod
    def _up_cat(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x_c4, x_c3, x_c2, x_c1):
        x43 = self.dec43(self._up_cat(x_c4, x_c3))
        x32 = self.dec32(self._up_cat(x43,  x_c2))
        x21 = self.dec21(self._up_cat(x32,  x_c1))
        out = self.seg_head(x21)

        if self.deep_supervision and self.training:
            return [out, self.seg_head_ds2(x32), self.seg_head_ds3(x43)]
        return out


class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__()

        hidden_size = c4_dims//factor
        c4_size = c4_dims
        c3_size = c4_dims//(factor**1)
        c2_size = c4_dims//(factor**2)
        c1_size = c4_dims//(factor**3)

        self.conv1_4 = nn.Conv2d(c4_size+c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)

    def forward(self, x_c4, x_c3, x_c2, x_c1):
        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)
        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)
        # fuse top-down features and Y1 features
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)), mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        return self.conv1_1(x)
