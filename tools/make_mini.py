# -*- coding: utf-8 -*-
"""生成合成迷你数据集（MPDD 风格目录），用于冒烟测试 train.py"""
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw

random.seed(42)
np.random.seed(42)

ROOT = Path("mini_data")


def make_normal(size=128):
    """灰色背景 + 轻微噪声 = 正常工件"""
    arr = np.full((size, size, 3), 140, dtype=np.int16)
    arr += np.random.randint(-15, 15, arr.shape).astype(np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_defect(size=128):
    """正常图 + 黑色块 = 缺陷；同时返回掩码图"""
    img = make_normal(size)
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    dm = ImageDraw.Draw(mask)
    x, y = random.randint(20, size - 40), random.randint(20, size - 40)
    r = random.randint(8, 15)
    d.ellipse([x - r, y - r, x + r, y + r], fill=(30, 30, 30))
    dm.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return img, mask


def save(imgs, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(imgs):
        im.save(folder / f"img_{i:03d}.png")


# train: 8 张正常
save([make_normal() for _ in range(8)], ROOT / "train" / "Part_A" / "good")
# test: 2 张正常
save([make_normal() for _ in range(2)], ROOT / "test" / "Part_A" / "good")
# test: 4 张缺陷 + 掩码
fdir = ROOT / "test" / "Part_A" / "scratch"
fdir.mkdir(parents=True, exist_ok=True)
for i in range(4):
    img, mask = make_defect()
    img.save(fdir / f"img_{i:03d}.png")
    mask.save(fdir / f"img_{i:03d}_mask.png")

for p in sorted(ROOT.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(ROOT)}")
print("mini dataset created:", ROOT)
