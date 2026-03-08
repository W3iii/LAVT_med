"""
data/dataset_bc.py
──────────────────
PyTorch Dataset for the BC (Breast Cancer) RIS dataset.

Annotation JSON format (per entry):
  {
    "image":     "BC001_S0022.png",
    "mask":      "BC001_S0022.png",
    "sentences": ["breast tumor lesion", ...],
    "is_pos":    1
  }

Images are single-channel uint8 PNGs (HU-windowed, soft tissue).
They are replicated to 3-channel RGB on-the-fly so the Swin backbone
receives the expected input without storing redundant disk data.

Masks are binary uint8 PNGs with values in {0, 1}.
"""

import os
import json
import random

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from bert.tokenization_bert import BertTokenizer


class BCDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.split            = split
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.max_tokens       = 20

        # ── load annotation JSON ──────────────────────────────────────────
        ann_path = os.path.join(args.bc_dataset_root, 'annotations', f'{split}.json')
        with open(ann_path, 'r') as f:
            self.annotations = json.load(f)

        self.image_dir = os.path.join(args.bc_dataset_root, 'images', split)
        self.mask_dir  = os.path.join(args.bc_dataset_root, 'masks',  split)

        # ── tokenise all sentences up-front ──────────────────────────────
        self.tokenizer    = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self.input_ids    = []
        self.attention_masks = []

        # pre-tokenise the empty string once; added to every sample's pool
        # so the model occasionally receives no language cue (robustness)
        _empty_ids    = [0] * self.max_tokens
        _empty_mask   = [0] * self.max_tokens
        _empty_tok    = self.tokenizer.encode(text='', add_special_tokens=True)
        _empty_tok    = _empty_tok[:self.max_tokens]
        _empty_ids[:len(_empty_tok)]  = _empty_tok
        _empty_mask[:len(_empty_tok)] = [1] * len(_empty_tok)
        self._empty_input_ids   = torch.tensor(_empty_ids).unsqueeze(0)
        self._empty_attn_mask   = torch.tensor(_empty_mask).unsqueeze(0)

        for ann in self.annotations:
            sentences_for_ref  = []
            attentions_for_ref = []

            for sentence_raw in ann['sentences']:
                padded_input_ids = [0] * self.max_tokens
                attention_mask   = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(
                    text=sentence_raw, add_special_tokens=True
                )
                input_ids = input_ids[:self.max_tokens]  # truncate

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)]   = [1] * len(input_ids)

                sentences_for_ref.append(
                    torch.tensor(padded_input_ids).unsqueeze(0)
                )
                attentions_for_ref.append(
                    torch.tensor(attention_mask).unsqueeze(0)
                )

            # append the empty-string token as an extra candidate
            sentences_for_ref.append(self._empty_input_ids)
            attentions_for_ref.append(self._empty_attn_mask)

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)

    # ── helpers ───────────────────────────────────────────────────────────

    def get_classes(self):
        return []

    def __len__(self):
        return len(self.annotations)

    # ── __getitem__ ───────────────────────────────────────────────────────

    def __getitem__(self, index):
        ann = self.annotations[index]

        # ── image: grayscale PNG → 3-channel RGB ─────────────────────────
        img_path = os.path.join(self.image_dir, ann['image'])
        img_gray = Image.open(img_path).convert('L')          # H×W uint8
        img      = Image.merge('RGB', [img_gray, img_gray, img_gray])

        # ── mask: binary PNG {0,1} kept as palette image ─────────────────
        mask_path = os.path.join(self.mask_dir, ann['mask'])
        mask = Image.open(mask_path)                           # values {0,1}
        # keep as mode "P" (palette) to match original ReferDataset contract
        mask = mask.convert('P')

        # ── joint spatial transforms ──────────────────────────────────────
        if self.image_transforms is not None:
            img, mask = self.image_transforms(img, mask)

        # ── sentence embedding: always pick one at random ─────────────────
        # The pool includes the empty string '' as the last entry, so the
        # model occasionally receives no language cue (robustness training).
        choice_sent       = np.random.choice(len(self.input_ids[index]))
        tensor_embeddings = self.input_ids[index][choice_sent]
        attention_mask    = self.attention_masks[index][choice_sent]

        return img, mask, tensor_embeddings, attention_mask
