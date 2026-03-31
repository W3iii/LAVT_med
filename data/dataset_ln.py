"""
data/dataset_ln.py
──────────────────
PyTorch Dataset for the LN (Lung Nodule) RIS dataset.

Annotation JSON format (per entry):
  {
    "image":      "CHEST1001_S0136.png",
    "mask":       "CHEST1001_S0136_cls1.png",   (or "" for negatives)
    "sentences":  ["benign lung nodule"],
    "is_pos":     1,                             (0 for negatives)
    "category":   1,                             (0 for normal slices)
    "patient_id": "CHEST1001"
  }

Images are single-channel uint8 PNGs (HU-windowed).
They are replicated to 3-channel RGB on-the-fly.

Masks are binary uint8 PNGs with values in {0, 1}.
When is_pos == 0, mask is "" → dataloader returns an empty (all-zero) mask.
"""

import os
import json
import random

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from bert.tokenization_bert import BertTokenizer


class LNDataset(data.Dataset):

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
        ann_path = os.path.join(args.ln_dataset_root, 'annotations', f'{split}.json')
        with open(ann_path, 'r') as f:
            self.annotations = json.load(f)

        self.image_dir = os.path.join(args.ln_dataset_root, 'images', split)
        self.mask_dir  = os.path.join(args.ln_dataset_root, 'masks',  split)

        # ── tokenise all sentences up-front ──────────────────────────────
        self.tokenizer       = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self.input_ids       = []
        self.attention_masks = []

        for ann in self.annotations:
            sentences_for_ref  = []
            attentions_for_ref = []

            for sentence_raw in ann['sentences']:
                padded_input_ids = [0] * self.max_tokens
                attention_mask   = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(
                    text=sentence_raw, add_special_tokens=True
                )
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)]   = [1] * len(input_ids)

                sentences_for_ref.append(
                    torch.tensor(padded_input_ids).unsqueeze(0)
                )
                attentions_for_ref.append(
                    torch.tensor(attention_mask).unsqueeze(0)
                )

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
        img_gray = Image.open(img_path).convert('L')
        img      = Image.merge('RGB', [img_gray, img_gray, img_gray])

        # ── mask ─────────────────────────────────────────────────────────
        if ann['is_pos'] == 1 and ann['mask']:
            mask_path = os.path.join(self.mask_dir, ann['mask'])
            mask = Image.open(mask_path).convert('P')
        else:
            # negative sample → empty mask (same size as image)
            w, h = img_gray.size
            mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode='P')

        # ── joint spatial transforms ──────────────────────────────────────
        if self.image_transforms is not None:
            img, mask = self.image_transforms(img, mask)

        # ── sentence embedding ────────────────────────────────────────────
        if self.split == 'train':
            choice_sent = np.random.choice(len(self.input_ids[index]))
        else:
            choice_sent = 0
        tensor_embeddings = self.input_ids[index][choice_sent]
        attention_mask    = self.attention_masks[index][choice_sent]

        return img, mask, tensor_embeddings, attention_mask
