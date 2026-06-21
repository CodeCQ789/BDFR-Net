import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FHPSDataset(Dataset):
    def __init__(self, root, split, img_size=256, augment=False, grayscale=False):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.grayscale = grayscale
        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"
        self.images = sorted(self.image_dir.glob("*.png"))
        self.masks = [self.mask_dir / p.name for p in self.images]
        missing = [p for p in self.masks if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} masks, first: {missing[0]}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(str(self.images[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.masks[index]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(self.masks[index])
        mask = np.where(mask == 1, 1, np.where(mask == 2, 2, 0)).astype(np.uint8)

        if self.augment:
            image, mask = self._augment(image, mask)

        if image.shape[:2] != (self.img_size, self.img_size):
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        if self.grayscale:
            image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            image = image[None, :, :]
            mean = np.array([0.5], dtype=np.float32)[:, None, None]
            std = np.array([0.229], dtype=np.float32)[:, None, None]
        else:
            image = image.transpose(2, 0, 1)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        image = (image - mean) / std

        sample = {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).long(),
            "name": self.images[index].name,
        }
        return sample

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        if random.random() < 0.5:
            angle = random.uniform(-20.0, 20.0)
            h, w = mask.shape
            mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            image = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if random.random() < 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-12.0, 12.0)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if random.random() < 0.5:
            gamma = random.uniform(0.8, 1.25)
            table = ((np.arange(256) / 255.0) ** gamma * 255.0).astype(np.uint8)
            image = cv2.LUT(image, table)

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)
