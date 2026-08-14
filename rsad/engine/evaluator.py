"""评估入口 — 加载 ckpt, 运行完整测试集评估"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.datasets import TestDataset, mask_collate
from ..data.transforms import test_transform
from ..metrics import compute_metrics


def evaluate(cfg):
    device = torch.device(cfg.device if cfg.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    from ..models.rsad_model import RSADModel
    from ..utils.checkpoint import load_ckpt

    model = RSADModel(cfg).to(device)
    load_ckpt(model, cfg.ckpt, device)
    model.eval()

    cls_filter = [cfg.cls] if getattr(cfg, "cls", None) else None
    ds = TestDataset(Path(cfg.data_root) / "test",
                     test_transform(cfg.crop if hasattr(cfg, "crop") else 224),
                     classes=cls_filter)
    loader = DataLoader(ds, batch_size=cfg.batch, shuffle=False,
                        num_workers=0, collate_fn=mask_collate)

    i_auc, p_auc = compute_metrics(model, loader, device)
    print(f"I-AUROC={i_auc:.4f} | P-AUROC={p_auc:.4f}")
    return i_auc, p_auc
