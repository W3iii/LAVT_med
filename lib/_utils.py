"""
lib/_utils.py  (v6)
─────────────────────────
Changes vs v5:
  * 移除 FiLM 的 beta path (additive cls→head shortcut，先前實驗證實會被學成 lookup)
  * 改用純乘性 sigmoid gating: attended * gate(cls_emb)
  * gate_proj 權重初始化為 0、bias=0 → 起步 gate=0.5 (恆等縮放)
  * 訓練必須讓 attended 帶有非常數視覺信號，否則整個 exist 輸出都會被乘成常數
"""

from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build the exist_module (shared by both classes)
# ─────────────────────────────────────────────────────────────────────────────
def _build_exist_module(c4_ch, c3_ch, cls_dim=768, hidden=256, dropout=0.3):
    """
    Class-conditioned attention pooling + pure multiplicative gating.
    No additive cls path — head input is strictly attended * gate(cls).
    """
    gate_proj = nn.Linear(cls_dim, hidden)
    nn.init.zeros_(gate_proj.weight)
    nn.init.zeros_(gate_proj.bias)

    return nn.ModuleDict({
        # project c4 / c3 spatial tokens to a unified hidden dim
        'c4_proj':    nn.Linear(c4_ch, hidden),
        'c3_proj':    nn.Linear(c3_ch, hidden),
        # query is built from class_embed, K/V from spatial tokens
        'query_proj': nn.Linear(cls_dim, hidden),
        'k_proj':     nn.Linear(hidden,  hidden),
        'v_proj':     nn.Linear(hidden,  hidden),
        # cls_emb → sigmoid gate over attended hidden dims
        'gate_proj':  gate_proj,
        # head only sees gated visual feature
        'head': nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        ),
    })


def _exist_attention_pool(exist_module, x_c4, x_c3, cls_emb):
    """
    Class-conditioned attention pooling over c3 + c4 spatial tokens.

    x_c4   : (B, c4_ch, H4, W4)
    x_c3   : (B, c3_ch, H3, W3)
    cls_emb: (B, cls_dim)        — class_embed lookup (before injection)

    Returns: logit (B,)
    """
    hidden = exist_module['k_proj'].out_features
    attn_scale = hidden ** -0.5

    # Flatten spatial → sequence, project to hidden
    c4_seq = x_c4.flatten(2).transpose(1, 2)             # (B, H4*W4, c4_ch)
    c3_seq = x_c3.flatten(2).transpose(1, 2)             # (B, H3*W3, c3_ch)
    c4_seq = exist_module['c4_proj'](c4_seq)             # (B, H4*W4, hidden)
    c3_seq = exist_module['c3_proj'](c3_seq)             # (B, H3*W3, hidden)
    feat_seq = torch.cat([c4_seq, c3_seq], dim=1)        # (B, N, hidden)

    # Class-conditioned query
    q = exist_module['query_proj'](cls_emb).unsqueeze(1) # (B, 1, hidden)
    k = exist_module['k_proj'](feat_seq)                 # (B, N, hidden)
    v = exist_module['v_proj'](feat_seq)                 # (B, N, hidden)

    # Scaled dot-product attention, single query
    attn = torch.matmul(q, k.transpose(1, 2)) * attn_scale   # (B, 1, N)
    attn = F.softmax(attn, dim=-1)
    attended = torch.matmul(attn, v).squeeze(1)              # (B, hidden)

    # Pure multiplicative gating: cls_emb scales attended per-dim, no additive path.
    # gate_proj is zero-init → gate=0.5 uniformly at t=0, so head sees attended/2.
    gate = torch.sigmoid(exist_module['gate_proj'](cls_emb))  # (B, hidden) ∈ [0,1]
    exist_feat = attended * gate                              # (B, hidden)
    return exist_module['head'](exist_feat).squeeze(-1)       # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# LAVT (external BERT)
# ─────────────────────────────────────────────────────────────────────────────
class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone   = backbone
        self.classifier = classifier

        # ── learnable class embedding ────────────────────────────────────
        self.class_embed     = nn.Embedding(5, 768)
        self.class_pos_embed = nn.Parameter(torch.zeros(1, 768, 1))
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.class_pos_embed,    std=0.02)

        self.class_gate = nn.Sequential(
            nn.Linear(768, 768),
            nn.Sigmoid(),
        )

        # ── existence head: class-conditioned attention pooling ─────────
        c4_ch = backbone.num_features[-1]
        c3_ch = backbone.num_features[-2]
        self.exist_module = _build_exist_module(c4_ch, c3_ch)

    def _inject_class_token(self, l_feats, l_mask, category):
        B = l_feats.shape[0]
        cls_emb      = self.class_embed(category)                 # (B, 768)
        gate         = self.class_gate(cls_emb)                   # (B, 768)
        gated_signal = (cls_emb * gate).unsqueeze(-1)             # (B, 768, 1)
        l_feats      = l_feats + gated_signal                     # broadcast
        cls_token    = cls_emb.unsqueeze(-1) + self.class_pos_embed  # (B, 768, 1)
        l_feats      = torch.cat([l_feats, cls_token], dim=-1)    # (B, 768, seq+1)
        cls_mask     = torch.ones(B, 1, 1, device=l_mask.device)
        l_mask       = torch.cat([l_mask, cls_mask], dim=1)       # (B, seq+1, 1)
        return l_feats, l_mask, cls_emb

    def forward(self, x, l_feats, l_mask, category=None):
        input_shape = x.shape[-2:]

        # Pre-compute cls_emb so exist head can use it directly
        if category is not None:
            l_feats, l_mask, cls_emb = self._inject_class_token(
                l_feats, l_mask, category)
        else:
            cls_emb = torch.zeros(x.shape[0], 768, device=x.device)

        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        seg = self.classifier(x_c4, x_c3, x_c2, x_c1)
        seg = F.interpolate(seg, size=input_shape, mode='bilinear',
                            align_corners=True)

        exist_out = _exist_attention_pool(
            self.exist_module, x_c4, x_c3, cls_emb)
        return seg, exist_out


class LAVT(_LAVTSimpleDecode):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# LAVT One: BERT inside the model
# ─────────────────────────────────────────────────────────────────────────────
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super().__init__()
        self.backbone     = backbone
        self.classifier   = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

        # ── learnable class embedding ────────────────────────────────────
        self.class_embed     = nn.Embedding(5, 768)
        self.class_pos_embed = nn.Parameter(torch.zeros(1, 768, 1))
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.class_pos_embed,    std=0.02)

        self.class_gate = nn.Sequential(
            nn.Linear(768, 768),
            nn.Sigmoid(),
        )

        # ── existence head: class-conditioned attention pooling ─────────
        c4_ch = backbone.num_features[-1]
        c3_ch = backbone.num_features[-2]
        self.exist_module = _build_exist_module(c4_ch, c3_ch)

    def _inject_class_token(self, l_feats, l_mask, category):
        B = l_feats.shape[0]
        cls_emb      = self.class_embed(category)
        gate         = self.class_gate(cls_emb)
        gated_signal = (cls_emb * gate).unsqueeze(-1)
        l_feats      = l_feats + gated_signal
        cls_token    = cls_emb.unsqueeze(-1) + self.class_pos_embed
        l_feats      = torch.cat([l_feats, cls_token], dim=-1)
        cls_mask     = torch.ones(B, 1, 1, device=l_mask.device)
        l_mask       = torch.cat([l_mask, cls_mask], dim=1)
        return l_feats, l_mask, cls_emb

    def forward(self, x, text, l_mask, category=None):
        input_shape = x.shape[-2:]
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]
        l_feats = l_feats.permute(0, 2, 1)
        l_mask  = l_mask.unsqueeze(dim=-1)

        if category is not None:
            l_feats, l_mask, cls_emb = self._inject_class_token(
                l_feats, l_mask, category)
        else:
            cls_emb = torch.zeros(x.shape[0], 768, device=x.device)

        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        seg = self.classifier(x_c4, x_c3, x_c2, x_c1)
        seg = F.interpolate(seg, size=input_shape, mode='bilinear',
                            align_corners=True)

        exist_out = _exist_attention_pool(
            self.exist_module, x_c4, x_c3, cls_emb)
        return seg, exist_out


class LAVTOne(_LAVTOneSimpleDecode):
    pass