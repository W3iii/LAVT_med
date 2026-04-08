from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        # existence head: predicts whether queried category exists in this slice
        feat_ch = classifier.feat_channels
        self.exist_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),   
            nn.Flatten(),
            nn.Linear(feat_ch * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        seg, feat = self.classifier(x_c4, x_c3, x_c2, x_c1, return_feat=True)
        seg = F.interpolate(seg, size=input_shape, mode='bilinear', align_corners=True)
        exist_out = self.exist_head(feat).squeeze(-1)  # (B,)

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
        feat_ch = classifier.feat_channels
        self.exist_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_ch, 1),
        )

    def forward(self, x, text, l_mask):
        input_shape = x.shape[-2:]
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (6, 10, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        seg, feat = self.classifier(x_c4, x_c3, x_c2, x_c1, return_feat=True)
        seg = F.interpolate(seg, size=input_shape, mode='bilinear', align_corners=True)
        exist_out = self.exist_head(feat).squeeze(-1)  # (B,)

        return seg, exist_out


class LAVTOne(_LAVTOneSimpleDecode):
    pass
