"""读取讲义提供的 OASIS PNG 数据，保留课程的 train/validate/test 划分。"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[2] / "data/oasis/keras_png_slices_data"
)


class OASISDataset(Dataset):
    def __init__(self, root, split, segmentation=False):
        if split not in ("train", "validate", "test"):
            raise ValueError("Use the course's train, validate or test split.")
        self.root = Path(root)
        self.split = split
        self.segmentation = segmentation
        self.files = sorted((self.root / f"keras_png_slices_{split}").glob("case_*.png"))
        if not self.files:
            raise FileNotFoundError(f"No course OASIS PNG images in {self.root}, split={split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as image:
            pixels = np.array(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(pixels).unsqueeze(0)
        if not self.segmentation:
            return x

        mask_path = self.root / f"keras_png_slices_seg_{self.split}" / path.name.replace(
            "case_", "seg_", 1
        )
        with Image.open(mask_path) as image:
            mask = np.array(image, dtype=np.int64)
        # 已检查课程所有标签 PNG：仅有 0、85、170、255，逐一映射为四类。
        if mask.shape != pixels.shape or not np.isin(mask, [0, 85, 170, 255]).all():
            raise ValueError(f"Unexpected OASIS mask shape or label values: {mask_path}")
        labels = torch.from_numpy(mask // 85)
        one_hot = F.one_hot(labels, num_classes=4).permute(2, 0, 1).float()
        return x, one_hot
