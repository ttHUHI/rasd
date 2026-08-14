"""数据加载: NormalTrainDataset / TestDataset / mask_collate（从 v1 迁移）

目录约定(MPDD):
  train/<类别>/good/*.jpg          → 训练正常样本
  test/<类别>/good/*.jpg           → 测试正常
  test/<类别>/<缺陷类型>/*.jpg + *_mask.png → 缺陷+掩码
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


class NormalTrainDataset(Dataset):
    """训练集——只加载正常样本。"""
    def __init__(self, root, transform):
        root = Path(root)
        self.files = []
        for ext in IMG_EXTS:
            self.files += list(root.rglob(ext))
        assert self.files, f"训练目录下没有图片: {root}"
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.transform(img)


class TestDataset(Dataset):
    """测试集——目录结构自动推断标签, 缺陷样本带掩码。"""
    def __init__(self, root, transform, mask_size=224):
        root = Path(root)
        self.items = []
        for ext in IMG_EXTS:
            for f in root.rglob(ext):
                if "mask" in f.stem.lower():
                    continue
                label = 0 if "good" in f.parts else 1
                mask = None
                if label == 1:
                    cands = [g for g in f.parent.glob("*mask*") if g.is_file()]
                    mask = cands[0] if cands else None
                self.items.append((f, label, mask))
        assert self.items, f"test 目录下没有图片: {root}"
        self.transform = transform
        from torchvision import transforms as T
        self.mask_transform = T.Compose([
            T.Resize((mask_size, mask_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, label, mask = self.items[i]
        img = Image.open(f).convert("RGB")
        x = self.transform(img)
        m = None
        if mask is not None:
            m = self.mask_transform(Image.open(mask).convert("L"))
            m = (m > 0.5).float()
        return x, label, m


def mask_collate(batch):
    """测试集 collate——正常样本掩码为 None, 自定义收集。"""
    imgs = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    masks = [b[2] for b in batch]
    return imgs, labels, masks
