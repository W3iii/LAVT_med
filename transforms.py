import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


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


class RandomBrightnessContrast(object):
    """
    Randomly adjust brightness and contrast of a PIL RGB image.
    Simulates soft-tissue windowing variation in CT scans.
    Applied before ToTensor.
    """
    def __init__(self, brightness=0.2, contrast=0.2, p=0.5):
        self.brightness = brightness
        self.contrast   = contrast
        self.p          = p

    def __call__(self, image, target):
        if random.random() < self.p:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image  = F.adjust_brightness(image, max(0.0, factor))
        if random.random() < self.p:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image  = F.adjust_contrast(image, max(0.0, factor))
        return image, target


class RandomGaussianNoise(object):
    """
    Add random Gaussian noise to a float tensor image.
    Applied after ToTensor, before Normalize.
    Simulates CT acquisition noise.
    """
    def __init__(self, std=0.02, p=0.5):
        self.std = std
        self.p   = p

    def __call__(self, image, target):
        if random.random() < self.p:
            noise = torch.randn_like(image) * self.std
            image = torch.clamp(image + noise, 0.0, 1.0)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image  = F.vflip(image)
            target = F.vflip(target)
        return image, target


class ForegroundCrop(object):
    """
    nnU-Net style foreground oversampling crop (for training).

    With probability `fg_prob`:
      - Find foreground pixels in target mask
      - Crop a patch centered on a random foreground pixel (with jitter)
    With probability `1 - fg_prob`:
      - Random crop

    If target has no foreground, always does random crop.
    Pads image/target if smaller than patch_size.
    Place BEFORE Resize in the transform pipeline.
    """
    def __init__(self, patch_size, fg_prob=0.67):
        self.patch_size = patch_size
        self.fg_prob = fg_prob

    def __call__(self, image, target):
        w, h = image.size  # PIL: (w, h)
        ps = self.patch_size

        # Pad if needed
        image = pad_if_smaller(image, ps, fill=0)
        target = pad_if_smaller(target, ps, fill=0)
        w, h = image.size

        # Check for foreground
        target_np = np.asarray(target)
        fg_coords = np.argwhere(target_np > 0)  # (N, 2) → (y, x)

        if len(fg_coords) > 0 and random.random() < self.fg_prob:
            # Foreground crop: pick a random fg pixel, crop around it
            idx = random.randint(0, len(fg_coords) - 1)
            cy, cx = fg_coords[idx]

            # Add jitter so nodule isn't always centered (±25% of patch)
            jitter = ps // 4
            cy += random.randint(-jitter, jitter)
            cx += random.randint(-jitter, jitter)

            # Compute crop box, clamp to image bounds
            y0 = max(0, min(cy - ps // 2, h - ps))
            x0 = max(0, min(cx - ps // 2, w - ps))
        else:
            # Random crop
            y0 = random.randint(0, max(0, h - ps))
            x0 = random.randint(0, max(0, w - ps))

        image = F.crop(image, y0, x0, ps, ps)
        target = F.crop(target, y0, x0, ps, ps)
        return image, target


def sliding_window_positions(h, w, patch_size, overlap=0.5):
    """
    Generate (y0, x0) positions for sliding window inference.
    Covers the entire image with overlapping patches.
    """
    step = int(patch_size * (1 - overlap))
    step = max(step, 1)
    positions = []
    for y0 in range(0, max(1, h - patch_size + 1), step):
        for x0 in range(0, max(1, w - patch_size + 1), step):
            positions.append((y0, x0))
    # Ensure bottom-right corner is covered
    if positions:
        last_y = max(0, h - patch_size)
        last_x = max(0, w - patch_size)
        if positions[-1] != (last_y, last_x):
            positions.append((last_y, last_x))
    else:
        positions.append((0, 0))
    return positions

