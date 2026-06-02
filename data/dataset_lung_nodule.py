import json
import random
from pathlib import Path

import numpy as np
import torch.utils.data as data
from PIL import Image


class LungNoduleDataset(data.Dataset):
    """
    2-class lung nodule slice dataset.

    Expects a directory layout produced by ``prepare_dataset.py``:
        data_root/
          images/{split}/*.png      uint8 grayscale (HU-windowed CT)
          masks/{split}/*.png       uint8 {0, 1} foreground mask
          annotations/{split}.json  list of {patient_id, image, mask}

    ``mask == "empty"`` denotes a normal slice (negative sample).

    Each epoch, ``resample_negatives(epoch)`` picks a fresh subset of
    negatives from the candidate pool, sized as
    ``len(positives) * neg_ratio``. Validation/test splits stay frozen by
    keeping the dataset at epoch 0.
    """

    def __init__(self, data_root, split, transforms,
                 neg_ratio: float = 0.3, seed: int = 42,
                 return_meta: bool = False):
        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms
        self.neg_ratio = neg_ratio
        self.return_meta = return_meta
        self.images_dir = self.data_root / "images" / split
        self.masks_dir = self.data_root / "masks" / split

        ann_path = self.data_root / "annotations" / f"{split}.json"
        with open(ann_path) as f:
            anns = json.load(f)

        self.positives = [a for a in anns if a["mask"] != "empty"]
        self.neg_pool = [a for a in anns if a["mask"] == "empty"]
        self._base_seed = seed
        self.samples: list = []
        self.resample_negatives(epoch=0)

    def resample_negatives(self, epoch: int) -> None:
        # neg_ratio < 0 -> use the full negative pool (test/eval mode).
        if self.neg_ratio < 0:
            self.samples = self.positives + list(self.neg_pool)
            return
        rng = random.Random(self._base_seed + epoch)
        n_neg = int(len(self.positives) * self.neg_ratio)
        n_neg = min(n_neg, len(self.neg_pool))
        sampled = rng.sample(self.neg_pool, n_neg) if n_neg > 0 else []
        self.samples = self.positives + sampled

    @property
    def patient_ids(self) -> list:
        return [s["patient_id"] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ann = self.samples[idx]
        img_path = self.images_dir / ann["image"]
        if img_path.suffix == ".npy":
            arr = np.load(img_path).astype(np.float32)  # (H, W) CTNorm float
            img = Image.fromarray(arr, mode="F")
        else:
            img = Image.open(img_path).convert("RGB")
        if ann["mask"] == "empty":
            target = Image.new("L", img.size, 0)
        else:
            target = Image.open(self.masks_dir / ann["mask"])
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        if self.return_meta:
            return img, target, ann
        return img, target
