import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, image, target):
        image = F.resize(image, (self.h, self.w))
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        target = F.resize(target, (self.h, self.w), interpolation=Image.NEAREST)
        return image, target


class ResizeWithPad(object):
    """Resize maintaining aspect ratio, then zero-pad to exactly (h, w)."""
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, image, target):
        orig_w, orig_h = image.size          # PIL size is (W, H)
        scale = min(self.h / orig_h, self.w / orig_w)
        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))

        image  = F.resize(image,  (new_h, new_w))
        target = F.resize(target, (new_h, new_w), interpolation=Image.NEAREST)

        pad_h = self.h - new_h              # pad bottom
        pad_w = self.w - new_w              # pad right
        # F.pad order: (left, top, right, bottom)
        image  = F.pad(image,  (0, 0, pad_w, pad_h), fill=0)
        target = F.pad(target, (0, 0, pad_w, pad_h), fill=0)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)  # Return a random integer N such that a <= N <= b. Alias for randrange(a, b+1)
        image = F.resize(image, size)
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.asarray(target).copy(), dtype=torch.int64)
        return image, target


class RandomAffine(object):
    def __init__(self, angle, translate, scale, shear, resample=0, fillcolor=None):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, image, target):
        affine_params = T.RandomAffine.get_params(self.angle, self.translate, self.scale, self.shear, image.size)
        image = F.affine(image, *affine_params)
        target = F.affine(target, *affine_params)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


# ---------------------------------------------------------------------------
# nnU-Net style augmentations
# ---------------------------------------------------------------------------

class RandomFlip(object):
    """Random horizontal flip."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image  = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image  = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomRotation(object):
    """Random rotation ±degrees."""
    def __init__(self, degrees=30, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            image  = F.rotate(image,  angle, fill=0)
            target = F.rotate(target, angle, interpolation=Image.NEAREST, fill=0)
        return image, target


class RandomScaleAndCrop(object):
    """
    Scale by a random factor then resize back to original size.
    scale > 1 → zoom in (random crop back).
    scale < 1 → zoom out (resize back, nodule appears smaller in frame).
    """
    def __init__(self, scale_range=(0.85, 1.25), p=0.5):
        self.lo, self.hi = scale_range
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        w0, h0 = image.size          # PIL (W, H)
        scale  = random.uniform(self.lo, self.hi)
        new_h  = int(round(h0 * scale))
        new_w  = int(round(w0 * scale))

        image  = F.resize(image,  (new_h, new_w))
        target = F.resize(target, (new_h, new_w), interpolation=Image.NEAREST)

        if new_h >= h0 and new_w >= w0:
            top  = random.randint(0, new_h - h0)
            left = random.randint(0, new_w - w0)
            image  = F.crop(image,  top, left, h0, w0)
            target = F.crop(target, top, left, h0, w0)
        else:
            image  = F.resize(image,  (h0, w0))
            target = F.resize(target, (h0, w0), interpolation=Image.NEAREST)

        return image, target


class GaussianNoise(object):
    """Add Gaussian noise to the image tensor; leaves the mask unchanged."""
    def __init__(self, std_range=(0.0, 0.1), p=0.15):
        self.lo, self.hi = std_range
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            std   = random.uniform(self.lo, self.hi)
            image = image + torch.randn_like(image) * std
        return image, target


class GammaAugmentation(object):
    """
    Gamma correction: shift to [0,1], apply x^gamma, shift back.
    Works for both z-score CT floats and [0,1] RGB tensors.
    """
    def __init__(self, gamma_range=(0.7, 1.5), p=0.3):
        self.lo, self.hi = gamma_range
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            gamma = random.uniform(self.lo, self.hi)
            mn  = image.min()
            rng = (image.max() - mn).clamp(min=1e-8)
            image = ((image - mn) / rng).pow(gamma) * rng + mn
        return image, target

