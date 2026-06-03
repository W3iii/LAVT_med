from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel
from bert.tokenization_bert import BertTokenizer


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: BERT + learnable prompt inside    #
###############################################
# Single-token adjectives used to seed the soft prompt embeddings.
# Each must tokenize to exactly one BERT wordpiece in bert-base-uncased.
_INIT_ADJECTIVES = [
    "small", "irregular", "suspicious", "abnormal",
    "round", "dense", "bright", "malignant",
]
_PREFIX_TEXT = "a slice of chest ct with"
_SUFFIX_TEXT = "lung nodule"


def _select_init_adjectives(n: int) -> list:
    if n <= len(_INIT_ADJECTIVES):
        return _INIT_ADJECTIVES[:n]
    # extra slots fall back to the strongest cue
    return _INIT_ADJECTIVES + ["abnormal"] * (n - len(_INIT_ADJECTIVES))


class _LAVTOneSimpleDecode(nn.Module):
    """
    Image-only forward: ``model(x)`` returns segmentation logits.

    The language branch is internal: a fixed text scaffold
    "[CLS] a slice of chest ct with [V1]...[Vn] lung nodule [SEP]"
    where [V1]..[Vn] are learnable soft-prompt parameters. Soft tokens
    are seeded from BERT hidden states of single-word adjectives placed
    in the same scaffold (see SPEC §2.4).
    """

    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

        n_soft = int(args.n_soft_tokens)
        self.n_soft_tokens = n_soft

        tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        prefix_ids = tokenizer.encode(_PREFIX_TEXT, add_special_tokens=False)
        suffix_ids = tokenizer.encode(_SUFFIX_TEXT, add_special_tokens=False)
        cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id

        adjs = _select_init_adjectives(n_soft)
        # Tokenize the whole adjective phrase and pad/truncate to exactly
        # n_soft tokens. Robust to cased tokenizers (e.g. BioBERT) that may
        # split a lowercase adjective into multiple wordpieces.
        adj_token_ids = tokenizer.encode(" ".join(adjs), add_special_tokens=False)
        if len(adj_token_ids) >= n_soft:
            adj_token_ids = adj_token_ids[:n_soft]
        else:
            pad_id = adj_token_ids[-1] if adj_token_ids else tokenizer.unk_token_id
            adj_token_ids = adj_token_ids + [pad_id] * (n_soft - len(adj_token_ids))

        prefix_full = [cls_id] + prefix_ids
        suffix_full = suffix_ids + [sep_id]
        seq_len = len(prefix_full) + n_soft + len(suffix_full)
        self.seq_len = seq_len

        self.register_buffer(
            "prefix_input_ids",
            torch.tensor(prefix_full, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "suffix_input_ids",
            torch.tensor(suffix_full, dtype=torch.long),
            persistent=False,
        )

        # Seed soft tokens from BERT hidden states at the adjective slots.
        full_ids = torch.tensor(prefix_full + adj_token_ids + suffix_full,
                                dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            self.text_encoder.eval()
            attn = torch.ones_like(full_ids)
            hidden = self.text_encoder(full_ids, attention_mask=attn)[0][0]
            soft_start = len(prefix_full)
            soft_init = hidden[soft_start:soft_start + n_soft].clone()
        self.text_encoder.train()
        self.soft_tokens = nn.Parameter(soft_init)

    def _get_language_features(self, batch_size: int):
        word_emb = self.text_encoder.embeddings.word_embeddings
        prefix_emb = word_emb(self.prefix_input_ids)        # (P, 768)
        suffix_emb = word_emb(self.suffix_input_ids)        # (S, 768)
        seq_emb = torch.cat([prefix_emb, self.soft_tokens, suffix_emb], dim=0)
        seq_emb = seq_emb.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        attn_mask = torch.ones(batch_size, self.seq_len,
                               dtype=torch.long, device=seq_emb.device)
        out = self.text_encoder(inputs_embeds=seq_emb, attention_mask=attn_mask)
        return out[0], attn_mask

    def forward(self, x):
        input_shape = x.shape[-2:]
        l_feats, l_mask = self._get_language_features(x.shape[0])
        l_feats = l_feats.permute(0, 2, 1)                  # (B, 768, L)
        l_mask  = l_mask.unsqueeze(-1)                      # (B, L, 1)
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        out = self.classifier(x_c4, x_c3, x_c2, x_c1)

        if isinstance(out, list):
            # Deep supervision: upsample only the primary (index 0) output to
            # full resolution; DS outputs stay at their native decoder resolution
            # so the loss can downsample the target to match them.
            out[0] = F.interpolate(out[0], size=input_shape,
                                   mode='bilinear', align_corners=True)
            return out

        return F.interpolate(out, size=input_shape,
                             mode='bilinear', align_corners=True)


class LAVTOne(_LAVTOneSimpleDecode):
    pass
