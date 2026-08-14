"""checkpoint 保存/加载"""
import torch
from pathlib import Path


def save_ckpt(model, out_dir, name, epoch, best, sigma):
    """保存可训练权重（extractor 冻结不变, 不冗余保存）+ 模型结构标志。"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ck = {
        "aligner": model.aligner.state_dict(),
        "disc": model.disc.state_dict(),
        "sigma": sigma,
        "epoch": epoch,
        "best": best,
        "feature_norm": bool(getattr(model.cfg, "feature_norm", False)),
    }
    path = Path(out_dir) / name
    torch.save(ck, path)
    return path


def load_ckpt(model, path, device):
    ck = torch.load(path, map_location=device)
    # extractor 冻结不保存; 兼容旧 ckpt 携带 extractor 的情况
    if "extractor" in ck:
        model.extractor.load_state_dict(ck["extractor"])
    model.aligner.load_state_dict(ck["aligner"])
    model.disc.load_state_dict(ck["disc"])
    # 结构标志写回 cfg, 保证评估/推理与训练行为一致
    if "feature_norm" in ck:
        model.cfg.feature_norm = bool(ck["feature_norm"])
    epoch = int(ck.get("epoch", 0))
    best = float(ck.get("best", 0.0))
    sigma = ck.get("sigma", 0.015)
    return epoch, best, sigma
