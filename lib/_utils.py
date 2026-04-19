"""
lib/_utils.py  (fixed v3)
─────────────────────────
Fix: exist_head built in __init__ using backbone.num_features.
No more lazy-build — exist_head params are in model.parameters() from the start.
"""

from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

        # ── learnable class embedding ────────────────────────────────────
        self.class_embed = nn.Embedding(5, 768)
        self.class_pos_embed = nn.Parameter(torch.zeros(1, 768, 1))
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.class_pos_embed, std=0.02)

        self.class_gate = nn.Sequential(
            nn.Linear(768, 768),
            nn.Sigmoid(),
        )

        # ── existence head (x_c4 + x_c3) ────────────────────────────────
        # backbone.num_features = [C, 2C, 4C, 8C]
        # swin tiny/small: [96, 192, 384, 768]
        # swin base:       [128, 256, 512, 1024]
        # swin large:      [192, 384, 768, 1536]
        c4_ch = backbone.num_features[-1]
        c3_ch = backbone.num_features[-2]
        self.exist_head = nn.Sequential(
            nn.Linear(c4_ch + c3_ch, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def _inject_class_token(self, l_feats, l_mask, category):
        B = l_feats.shape[0]
        cls_emb = self.class_embed(category)                      # (B, 768)
        gate = self.class_gate(cls_emb)                           # (B, 768)
        gated_signal = (cls_emb * gate).unsqueeze(-1)             # (B, 768, 1)
        l_feats = l_feats + gated_signal                          # broadcast
        cls_token = cls_emb.unsqueeze(-1) + self.class_pos_embed  # (B, 768, 1)
        l_feats = torch.cat([l_feats, cls_token], dim=-1)         # (B, 768, seq+1)
        cls_mask = torch.ones(B, 1, 1, device=l_mask.device)
        l_mask = torch.cat([l_mask, cls_mask], dim=1)             # (B, seq+1, 1)
        return l_feats, l_mask

    def _exist_forward(self, x_c4, x_c3):
        c4_pool = F.adaptive_avg_pool2d(x_c4, 1).flatten(1)      # (B, c4_ch)
        c3_pool = F.adaptive_avg_pool2d(x_c3, 1).flatten(1)      # (B, c3_ch)
        exist_feat = torch.cat([c4_pool, c3_pool], dim=1)         # (B, c4+c3)
        return self.exist_head(exist_feat).squeeze(-1)             # (B,)

    def forward(self, x, l_feats, l_mask, category=None):
        input_shape = x.shape[-2:]
        if category is not None:
            l_feats, l_mask = self._inject_class_token(l_feats, l_mask, category)
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        seg = self.classifier(x_c4, x_c3, x_c2, x_c1)
        seg = F.interpolate(seg, size=input_shape, mode='bilinear',
                            align_corners=True)
        exist_out = self._exist_forward(x_c4, x_c3)
        return seg, exist_out


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

        # ── learnable class embedding ────────────────────────────────────
        self.class_embed = nn.Embedding(5, 768)
        self.class_pos_embed = nn.Parameter(torch.zeros(1, 768, 1))
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.class_pos_embed, std=0.02)

        self.class_gate = nn.Sequential(
            nn.Linear(768, 768),
            nn.Sigmoid(),
        )

        # ── existence head (x_c4 + x_c3) ────────────────────────────────
        c4_ch = backbone.num_features[-1]
        c3_ch = backbone.num_features[-2]
        self.exist_head = nn.Sequential(
            nn.Linear(c4_ch + c3_ch, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def _inject_class_token(self, l_feats, l_mask, category):
        B = l_feats.shape[0]
        cls_emb = self.class_embed(category)
        gate = self.class_gate(cls_emb)
        gated_signal = (cls_emb * gate).unsqueeze(-1)
        l_feats = l_feats + gated_signal
        cls_token = cls_emb.unsqueeze(-1) + self.class_pos_embed
        l_feats = torch.cat([l_feats, cls_token], dim=-1)
        cls_mask = torch.ones(B, 1, 1, device=l_mask.device)
        l_mask = torch.cat([l_mask, cls_mask], dim=1)
        return l_feats, l_mask

    def _exist_forward(self, x_c4, x_c3):
        c4_pool = F.adaptive_avg_pool2d(x_c4, 1).flatten(1)
        c3_pool = F.adaptive_avg_pool2d(x_c3, 1).flatten(1)
        exist_feat = torch.cat([c4_pool, c3_pool], dim=1)
        return self.exist_head(exist_feat).squeeze(-1)

    def forward(self, x, text, l_mask, category=None):
        input_shape = x.shape[-2:]
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]
        l_feats = l_feats.permute(0, 2, 1)
        l_mask = l_mask.unsqueeze(dim=-1)
        if category is not None:
            l_feats, l_mask = self._inject_class_token(l_feats, l_mask, category)
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        seg = self.classifier(x_c4, x_c3, x_c2, x_c1)
        seg = F.interpolate(seg, size=input_shape, mode='bilinear',
                            align_corners=True)
        exist_out = self._exist_forward(x_c4, x_c3)
        return seg, exist_out


class LAVTOne(_LAVTOneSimpleDecode):
    pass