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

from transformers import BertTokenizer


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
        self.iters_per_epoch  = getattr(args, 'iters_per_epoch', 0) if split == 'train' else 0
        self.fg_prob          = getattr(args, 'fg_prob', 0.67)

        # ── load annotation JSON ──────────────────────────────────────────
        ann_path = os.path.join(args.ln_dataset_root, 'annotations', f'{split}.json')
        with open(ann_path, 'r') as f:
            all_annotations = json.load(f)

        # ── negative sampling config ──────────────────────────────────────
        self.neg_ratio = getattr(args, 'neg_ratio', 2.0)
        self._pos = [a for a in all_annotations if a['is_pos'] == 1]
        self._neg = [a for a in all_annotations if a['is_pos'] == 0]

        if split == 'train' and self.neg_ratio > 0:
            self.resample_negatives(epoch=0)
        else:
            self.annotations = all_annotations

        self.image_dir = os.path.join(args.ln_dataset_root, 'images', split)
        self.mask_dir  = os.path.join(args.ln_dataset_root, 'masks',  split)

        # ── tokeniser (cache by unique sentence) ─────────────────────────
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self._token_cache = {}  # sentence_str → (input_ids_tensor, attn_mask_tensor)
        self._build_token_cache(all_annotations)

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_token_cache(self, all_annotations):
        """Tokenise each unique sentence once and cache the result."""
        for ann in all_annotations:
            for s in ann['sentences']:
                if s in self._token_cache:
                    continue
                padded = [0] * self.max_tokens
                attn   = [0] * self.max_tokens
                ids = self.tokenizer.encode(text=s, add_special_tokens=True)
                ids = ids[:self.max_tokens]
                padded[:len(ids)] = ids
                attn[:len(ids)]   = [1] * len(ids)
                self._token_cache[s] = (
                    torch.tensor(padded).unsqueeze(0),
                    torch.tensor(attn).unsqueeze(0),
                )
        print(f'  Token cache: {len(self._token_cache)} unique sentences')

    def resample_negatives(self, epoch=0):
        """Re-sample negative subset each epoch so all negatives are seen over time."""
        max_neg = int(len(self._pos) * self.neg_ratio)
        if len(self._neg) > max_neg:
            rng = random.Random(42 + epoch)
            sampled_neg = rng.sample(self._neg, max_neg)
        else:
            sampled_neg = self._neg
        self.annotations = self._pos + sampled_neg
        random.Random(42 + epoch).shuffle(self.annotations)
        print(f'  [train] resample epoch {epoch}: pos={len(self._pos)}, '
              f'neg={len(sampled_neg)}/{len(self._neg)}, '
              f'total={len(self.annotations)}')

    def get_classes(self):
        return []

    def __len__(self):
        if self.iters_per_epoch > 0:
            return self.iters_per_epoch
        return len(self.annotations)

    # ── __getitem__ ───────────────────────────────────────────────────────

    def __getitem__(self, index):
        if self.iters_per_epoch > 0:
            # nnU-Net style: randomly pick a sample each iteration
            # fg_prob chance to pick a positive sample
            if self._pos and random.random() < self.fg_prob:
                ann = random.choice(self._pos)
            else:
                ann = random.choice(self.annotations)
        else:
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

        # ── sentence embedding (from cache) ──────────────────────────────
        sentences = ann['sentences']
        if self.split == 'train':
            choice_sent = np.random.choice(len(sentences))
        else:
            choice_sent = 0
        tensor_embeddings, attention_mask = self._token_cache[sentences[choice_sent]]

        meta = {
            'is_pos':   ann['is_pos'],
            'category': ann.get('category', -1),
        }

        return img, mask, tensor_embeddings, attention_mask, meta
