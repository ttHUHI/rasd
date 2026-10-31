"""rsad.inference — 轻量推理：单图/目录 → 图像级异常分数 + 异常热图。

用法（CLI 见 tools/predict.py）:
    python tools/predict.py --config configs/mpdd.yaml \
        --ckpt checkpoints/best.pt --input img.png [--vis vis_out]
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..data.transforms import test_transform
from ..models.rsad_model import RSADModel
from ..utils.checkpoint import load_ckpt

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def load_model(cfg, ckpt, device):
    """构建 RSADModel 并加载 checkpoint（eval 模式）。"""
    model = RSADModel(cfg).to(device)
    load_ckpt(model, ckpt, device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model, img_path, transform, device):
    """单张图片推理，返回 (s_ad, anom_224, img)。

    s_ad:     图像级异常分数（异常图最大值，论文 §3.4）
    anom_224: 224×224 异常热图 (numpy, 0~1)
    img:      原始 PIL 图
    """
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    prob = model.test_forward(x)                            # (1,1,28,28) 正常概率
    anom = 1.0 - prob                                       # 异常图
    s_ad = anom.amax().item()
    anom_224 = F.interpolate(anom, size=(224, 224), mode="bilinear",
                             align_corners=False).squeeze().cpu().numpy()
    return s_ad, anom_224, img


def save_heatmap(img, anom_224, save_path, s_ad=None):
    """原图 + 异常热图叠加，保存到 save_path。"""
    base = img.resize((224, 224), Image.BILINEAR)
    base_np = np.asarray(base).astype(np.float32)
    cm = np.zeros((224, 224, 3), dtype=np.uint8)
    cm[:, :, 2] = np.clip(anom_224 * 4.0 * 255, 0, 255)          # 低 → 蓝
    cm[:, :, 0] = np.clip((anom_224 - 0.25) * 4.0 * 255, 0, 255)  # 高 → 红
    heat = np.clip(base_np * 0.55 + cm.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    out = Image.fromarray(heat)
    if s_ad is not None:
        from PIL import ImageDraw
        d = ImageDraw.Draw(out)
        d.text((5, 5), f"S_AD={s_ad:.3f}", fill=(255, 255, 0))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(save_path)


def collect_images(path):
    """输入文件或目录 → 图片路径列表（跳过 *_mask*）。"""
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*")
                  if f.suffix.lower() in IMG_EXT and "mask" not in f.stem.lower())
