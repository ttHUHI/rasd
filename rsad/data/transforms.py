"""图像预处理: 256 resize + 224 中心裁剪 + ImageNet 归一化（论文 §4.3）

可选对比度增强（contrast='clahe'）: 深色表面划痕对比度低, CLAHE 局部直方图
均衡放大低对比缺陷, 提高 WRN50 早期层响应（默认 None 保持论文原样）。
训练与推理必须使用相同设置（contrast 存入 checkpoint 自动一致）。
"""
import numpy as np
import cv2
from torchvision import transforms as T
from PIL import Image

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def _apply_contrast(img, mode):
    """对比度增强（RGB PIL → RGB PIL）。mode: 'clahe' / None。"""
    if not mode:
        return img
    arr = np.asarray(img)                       # (H,W,3) uint8
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def train_transform(crop=224, contrast=None):
    """训练预处理（contrast 默认 None, 保持论文原样）。"""
    ops = []
    if contrast:
        ops.append(T.Lambda(lambda img: _apply_contrast(img, contrast)))
    ops += [T.Resize(256), T.CenterCrop(crop), T.ToTensor(), T.Normalize(IMG_MEAN, IMG_STD)]
    return T.Compose(ops)


def test_transform(crop=224, contrast=None):
    # 与训练一致（含对比度设置, 保证评估/推理与训练相同输入分布）
    return train_transform(crop, contrast=contrast)
