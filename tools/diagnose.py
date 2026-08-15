# -*- coding: utf-8 -*-
"""训练收敛诊断：3 epoch 短训，打印 loss 分量 / pos-neg logit 分布 /
梯度范数 / 特征尺度（验证 sigma=0.015 噪声相对特征量级的可判别性）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from rsad.utils.config import load_config
from rsad.utils.seed import set_seed
from rsad.data.datasets import NormalTrainDataset, TestDataset, mask_collate
from rsad.data.transforms import train_transform, test_transform
from rsad.models.rsad_model import RSADModel
from rsad.losses import rsad_loss
from rsad.metrics import compute_metrics

set_seed()
cfg = load_config("configs/mpdd.yaml")
# 对照实验覆盖: --sigma --gamma --alpha [--feature_norm] [--epochs]
args = {k: v for k, v in (a.split("=") for a in sys.argv[1:])}
if "sigma" in args:
    cfg.sigma = float(args["sigma"])
if "gamma" in args:
    cfg.focal_gamma = float(args["gamma"])
if "alpha" in args:
    cfg.focal_alpha = float(args["alpha"])
if "feature_norm" in args:
    cfg.feature_norm = args["feature_norm"].lower() in ("1", "true", "yes")
if "noise_smooth" in args:
    import re
    m = re.fullmatch(r"k(\d+)s([\d.]+)", args["noise_smooth"].strip())
    cfg.noise_smooth = ({"kernel": int(m.group(1)), "sigma": float(m.group(2))}
                        if m else None)
EPOCHS = int(args.get("epochs", 3))
print(f"[diag] sigma={cfg.sigma} gamma={cfg.focal_gamma} alpha={cfg.focal_alpha} "
      f"feature_norm={getattr(cfg, 'feature_norm', False)} "
      f"noise_smooth={getattr(cfg, 'noise_smooth', None)} epochs={EPOCHS}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = Path(cfg.data_root)
train_ds = NormalTrainDataset(root / "train", train_transform(cfg.crop))
test_ds = TestDataset(root / "test", test_transform(cfg.crop))
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=mask_collate)

model = RSADModel(cfg).to(device)
opt = optim.Adam([
    {"params": model.aligner.parameters(), "lr": cfg.lr},
    {"params": model.disc.parameters(), "lr": cfg.lr_d},
], weight_decay=cfg.wd if hasattr(cfg, "wd") else 1e-5)


def grad_norm(named_params):
    tot = 0.0
    for _, p in named_params:
        if p.grad is not None:
            tot += p.grad.data.norm(2).item() ** 2
    return tot ** 0.5


# 特征尺度（aligned 输出的量级，用于评估 sigma 噪声相对强度）
with torch.no_grad():
    probe = next(iter(train_loader)).to(device)
    feats = model.extractor(probe)
    aligned = model.aligner(feats)
    print(f"[probe] feats std={feats.std():.4f} aligned std={aligned.std():.4f} "
          f"| noise(sigma={cfg.sigma}) 相对 feats 量级 = {cfg.sigma / feats.std():.2e}")

for ep in range(1, EPOCHS + 1):
    model.extractor.eval()
    model.aligner.train()
    model.disc.train()
    tot_l = tot_s = tot_c = 0.0
    pls, nls = [], []
    g_a = g_d = 0.0
    n = len(train_loader.dataset)
    for imgs in train_loader:
        imgs = imgs.to(device)
        pl, nl = model.train_forward(imgs)
        loss, ls, lc = rsad_loss(pl, nl, th_pos=cfg.th_pos, th_neg=cfg.th_neg,
                                 gamma=cfg.focal_gamma, alpha=cfg.focal_alpha)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot_l += loss.item() * len(imgs)
        tot_s += ls.item() * len(imgs)
        tot_c += lc.item() * len(imgs)
        pls.append(pl.detach())
        nls.append(nl.detach())
        g_a = max(g_a, grad_norm(model.aligner.named_parameters()))
        g_d = max(g_d, grad_norm(model.disc.named_parameters()))
    pl = torch.cat(pls)
    nl = torch.cat(nls)
    print(f"ep{ep}: loss={tot_l / n:.4f} seg={tot_s / n:.4f} cls={tot_c / n:.4f} "
          f"pos_logit mean={pl.mean():.3f} std={pl.std():.3f} "
          f"neg_logit mean={nl.mean():.3f} std={nl.std():.3f} "
          f"|g_aligner|={g_a:.4f} |g_disc|={g_d:.4f}")
    model.eval()
    i_auc, p_auc = compute_metrics(model, test_loader, device)
    print(f"  -> I-AUROC={i_auc:.4f} P-AUROC={p_auc:.4f}")
