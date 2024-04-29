import os
from typing import Optional
from enum import Enum, auto

import albumentations as A
from PIL import Image
from torch.utils.data import Dataset


class GuidanceType(Enum):
    CLS = auto()
    CLS_FREE = auto()
    REG = auto()


class ImgDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        img_size: tuple[int, int] = (32, 32),
        cls_guidance: GuidanceType = GuidanceType.REG,
        transform: Optional[A.Compose] = None,
    ):
        self.data_path = data_path
        self.imgs = os.listdir(data_path)
        self.transform = transform

        self._resize = A.Compose(
            [
                A.Resize(*img_size),
            ]
        )

    def __len__(self):
        return len(self.imgs)

    def _preprocess(self, img):
        img = self._resize(image=img)["image"]
        return img / 255.0

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = self._preprocess(img)

        if self.transform:
            img = self.transform(img)

        return img
