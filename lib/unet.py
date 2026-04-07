"""
lib/unet.py
───────────
Text-conditioned U-Net with swappable encoder for referring image segmentation.

Encoder options:
  - 'plain': standard conv blocks (lightweight, no pretrained weights)
  - 'resnet50':  torchvision ResNet-50  (ImageNet pretrained)
  - 'resnet101': torchvision ResNet-101 (ImageNet pretrained)

Text conditioning via PWAM at each encoder stage,
reusing the same PWAM / SpatialImageLanguageAttention from LAVT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .backbone import PWAM


# ── Encoder blocks ────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two 3x3 convs + BN + ReLU (standard U-Net block)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample + concat skip + ConvBlock."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── Plain encoder ─────────────────────────────────────────────────────────────

class PlainEncoder(nn.Module):
    """4-stage plain conv encoder."""
    out_channels = [64, 128, 256, 512]

    def __init__(self, in_ch=3):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)             # /1
        e2 = self.enc2(self.pool(e1))  # /2
        e3 = self.enc3(self.pool(e2))  # /4
        e4 = self.enc4(self.pool(e3))  # /8
        return [e1, e2, e3, e4]


# ── ResNet encoder ────────────────────────────────────────────────────────────

class ResNetEncoder(nn.Module):
    """
    4-stage ResNet encoder (ResNet-50 or ResNet-101).
    Outputs feature maps at /1 (via learned upsample), /4, /8, /16.
    Adjusted to provide a /1 resolution feature via a shallow head.
    """
    def __init__(self, depth=50, pretrained=True):
        super().__init__()
        if depth == 50:
            resnet = tv_models.resnet50(pretrained=pretrained)
        elif depth == 101:
            resnet = tv_models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f'Unsupported ResNet depth: {depth}')

        # stage0: conv1 + bn1 + relu (stride 2) → /2
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool  # /4

        # stage1-4
        self.layer1 = resnet.layer1  # /4,  256 ch
        self.layer2 = resnet.layer2  # /8,  512 ch
        self.layer3 = resnet.layer3  # /16, 1024 ch
        self.layer4 = resnet.layer4  # /32, 2048 ch

        # shallow head to produce /1 resolution feature (for small object preservation)
        self.shallow = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.out_channels = [64, 256, 512, 1024]

    def forward(self, x):
        e1 = self.shallow(x)                    # /1, 64 ch
        s  = self.stem(x)                        # /2, 64 ch
        s  = self.pool(s)                        # /4
        e2 = self.layer1(s)                      # /4, 256 ch
        e3 = self.layer2(e2)                     # /8, 512 ch
        e4 = self.layer3(e3)                     # /16, 1024 ch
        return [e1, e2, e3, e4]


# ── Text-conditioned U-Net ────────────────────────────────────────────────────

class TextConditionedUNet(nn.Module):
    """
    U-Net with PWAM text conditioning at each encoder stage.

    Args:
        encoder: one of PlainEncoder / ResNetEncoder
        lang_dim: language feature dimension (768 for BERT)
        num_heads_fusion: number of attention heads in PWAM per stage
        fusion_drop: dropout in PWAM
    """
    def __init__(self, encoder, lang_dim=768, num_heads_fusion=None, fusion_drop=0.0):
        super().__init__()
        self.backbone = encoder  # named 'backbone' for compatibility with train.py param grouping
        chs = encoder.out_channels  # [c1, c2, c3, c4]

        if num_heads_fusion is None:
            num_heads_fusion = [1, 1, 1, 1]

        # PWAM + gated residual at each encoder stage (same as LAVT Swin stages)
        # Stage 1: PWAM runs on /4 downsampled feature to save memory,
        #          then upsampled back to /1 for full-resolution text conditioning
        self.pwam1_downsample = 4  # downsample factor for stage 1 PWAM
        self.pwam1 = PWAM(chs[0], chs[0], lang_dim, chs[0], chs[0],
                          num_heads=num_heads_fusion[0], dropout=fusion_drop)
        self.pwam2 = PWAM(chs[1], chs[1], lang_dim, chs[1], chs[1],
                          num_heads=num_heads_fusion[1], dropout=fusion_drop)
        self.pwam3 = PWAM(chs[2], chs[2], lang_dim, chs[2], chs[2],
                          num_heads=num_heads_fusion[2], dropout=fusion_drop)
        self.pwam4 = PWAM(chs[3], chs[3], lang_dim, chs[3], chs[3],
                          num_heads=num_heads_fusion[3], dropout=fusion_drop)

        # gated residual (matches LAVT backbone's res_gate)
        self.res_gate1 = nn.Sequential(nn.Linear(chs[0], chs[0], bias=False), nn.ReLU(),
                                       nn.Linear(chs[0], chs[0], bias=False), nn.Tanh())
        self.res_gate2 = nn.Sequential(nn.Linear(chs[1], chs[1], bias=False), nn.ReLU(),
                                       nn.Linear(chs[1], chs[1], bias=False), nn.Tanh())
        self.res_gate3 = nn.Sequential(nn.Linear(chs[2], chs[2], bias=False), nn.ReLU(),
                                       nn.Linear(chs[2], chs[2], bias=False), nn.Tanh())
        self.res_gate4 = nn.Sequential(nn.Linear(chs[3], chs[3], bias=False), nn.ReLU(),
                                       nn.Linear(chs[3], chs[3], bias=False), nn.Tanh())

        # bottleneck
        self.bottleneck = ConvBlock(chs[3], chs[3] * 2)
        self.pool = nn.MaxPool2d(2)

        # decoder
        self.dec4 = DecoderBlock(chs[3] * 2 + chs[3], chs[3])
        self.dec3 = DecoderBlock(chs[3] + chs[2], chs[2])
        self.dec2 = DecoderBlock(chs[2] + chs[1], chs[1])
        self.dec1 = DecoderBlock(chs[1] + chs[0], chs[0])

        # output: named 'classifier' for compatibility with train.py param grouping
        self.classifier = nn.Conv2d(chs[0], 2, 1)

    def forward(self, x, l_feats, l_mask):
        """
        x:       (B, 3, H, W) image
        l_feats: (B, 768, N_l) language features
        l_mask:  (B, N_l, 1)   language attention mask
        """
        input_shape = x.shape[-2:]

        # encoder
        features = self.backbone(x)  # [e1, e2, e3, e4]
        e1, e2, e3, e4 = features

        # PWAM text injection + gated residual at each stage
        # Stage 1: downsample → PWAM → upsample to avoid OOM on /1 resolution
        e1 = self._apply_pwam_downsampled(self.pwam1, self.res_gate1, e1, l_feats, l_mask,
                                          ds_factor=self.pwam1_downsample)
        e2 = self._apply_pwam(self.pwam2, self.res_gate2, e2, l_feats, l_mask)
        e3 = self._apply_pwam(self.pwam3, self.res_gate3, e3, l_feats, l_mask)
        e4 = self._apply_pwam(self.pwam4, self.res_gate4, e4, l_feats, l_mask)

        # bottleneck
        b = self.bottleneck(self.pool(e4))

        # decoder with skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.classifier(d1)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=True)
        return out

    def _apply_pwam(self, pwam, res_gate, feat, l_feats, l_mask):
        """
        Apply PWAM with gated residual, matching LAVT backbone:
            x_residual = pwam(x, l, l_mask)
            x = x + res_gate(x_residual) * x_residual
        """
        B, C, H, W = feat.shape
        feat_flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H*W, C)
        x_residual = pwam(feat_flat, l_feats, l_mask)               # (B, H*W, C)
        feat_flat = feat_flat + res_gate(x_residual) * x_residual   # gated residual
        return feat_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)   # (B, C, H, W)

    def _apply_pwam_downsampled(self, pwam, res_gate, feat, l_feats, l_mask, ds_factor=4):
        """
        Stage 1 PWAM: downsample /1 feature → run PWAM at /ds_factor → upsample residual back.
        Keeps full /1 resolution in skip connection while adding text conditioning.
        480×480 → 120×120 (14,400 tokens) for PWAM → upsample gate back to 480×480.
        """
        B, C, H, W = feat.shape
        Hd, Wd = H // ds_factor, W // ds_factor

        # downsample feature for PWAM
        feat_down = F.adaptive_avg_pool2d(feat, (Hd, Wd))                          # (B, C, Hd, Wd)
        feat_down_flat = feat_down.permute(0, 2, 3, 1).reshape(B, Hd * Wd, C)      # (B, Hd*Wd, C)

        # PWAM on downsampled feature
        x_residual_down = pwam(feat_down_flat, l_feats, l_mask)                     # (B, Hd*Wd, C)
        gate_down = res_gate(x_residual_down) * x_residual_down                     # (B, Hd*Wd, C)

        # upsample gate back to original resolution
        gate_2d = gate_down.reshape(B, Hd, Wd, C).permute(0, 3, 1, 2)              # (B, C, Hd, Wd)
        gate_up = F.interpolate(gate_2d, size=(H, W), mode='bilinear', align_corners=True)  # (B, C, H, W)

        return feat + gate_up
