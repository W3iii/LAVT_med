import torch
import torch.nn as nn
from .backbone import PWAM


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, bias=True)
        self.norm = nn.InstanceNorm2d(out_ch, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class PlainConvStage(nn.Module):
    """Two ConvNormAct blocks; first one applies the given stride."""
    def __init__(self, in_ch, out_ch, stride, n_convs=2):
        super().__init__()
        layers = [ConvNormAct(in_ch, out_ch, stride=stride)]
        for _ in range(n_convs - 1):
            layers.append(ConvNormAct(out_ch, out_ch))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


class MultiModalNnUNetBackbone(nn.Module):
    """
    nnUNet 2D PlainConv encoder (6 stages) with PWAM language fusion.

    Stages 0-1  : stem, no PWAM
    Stages 2-5  : output c1..c4 with PWAM

    Cumulative strides (288×384 input example):
      stage 0  ×1  → 288×384   32 ch
      stage 1  ×2  → 144×192   64 ch
      stage 2  ×4  →  72×96   128 ch  → c1
      stage 3  ×8  →  36×48   256 ch  → c2
      stage 4  ×16 →  18×24   512 ch  → c3
      stage 5  ×32 →   9×12   512 ch  → c4

    Returns: tuple (c1, c2, c3, c4) as (B, C, H, W) tensors.
    out_channels attribute: [128, 256, 512, 512]
    """

    FEATURES = [32, 64, 128, 256, 512, 512]
    STRIDES  = [1,  2,   2,   2,   2,   2]

    def __init__(self, in_chans=1, num_heads_fusion=(1, 1, 1, 1), fusion_drop=0.0):
        super().__init__()
        feats   = self.FEATURES
        strides = self.STRIDES

        self.stage0 = PlainConvStage(in_chans, feats[0], stride=strides[0])
        self.stage1 = PlainConvStage(feats[0],  feats[1], stride=strides[1])
        self.stage2 = PlainConvStage(feats[1],  feats[2], stride=strides[2])
        self.stage3 = PlainConvStage(feats[2],  feats[3], stride=strides[3])
        self.stage4 = PlainConvStage(feats[3],  feats[4], stride=strides[4])
        self.stage5 = PlainConvStage(feats[4],  feats[5], stride=strides[5])

        fuse_dims = [feats[2], feats[3], feats[4], feats[5]]  # 128, 256, 512, 512
        self.fusions  = nn.ModuleList()
        self.res_gates = nn.ModuleList()
        for dim, nhead in zip(fuse_dims, num_heads_fusion):
            self.fusions.append(
                PWAM(dim, dim, 768, dim, dim, num_heads=nhead, dropout=fusion_drop)
            )
            gate = nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.ReLU(),
                nn.Linear(dim, dim, bias=False),
                nn.Tanh(),
            )
            nn.init.zeros_(gate[0].weight)
            nn.init.zeros_(gate[2].weight)
            self.res_gates.append(gate)

        self.out_channels = fuse_dims  # [128, 256, 512, 512]

    def _fuse(self, x, l, l_mask, fusion, res_gate):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)    # (B, H*W, C)
        x_res  = fusion(x_flat, l, l_mask)        # (B, H*W, C)
        x_flat = x_flat + res_gate(x_res) * x_res
        return x_flat.transpose(1, 2).view(B, C, H, W)

    def forward(self, x, l, l_mask):
        x  = self.stage0(x)
        x  = self.stage1(x)

        x  = self.stage2(x)
        c1 = self._fuse(x, l, l_mask, self.fusions[0], self.res_gates[0])

        x  = self.stage3(c1)
        c2 = self._fuse(x, l, l_mask, self.fusions[1], self.res_gates[1])

        x  = self.stage4(c2)
        c3 = self._fuse(x, l, l_mask, self.fusions[2], self.res_gates[2])

        x  = self.stage5(c3)
        c4 = self._fuse(x, l, l_mask, self.fusions[3], self.res_gates[3])

        return c1, c2, c3, c4
