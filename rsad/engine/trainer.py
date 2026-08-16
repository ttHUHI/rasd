"""训练循环 — 论文式自身加噪(aligned.detach(), 无队列)"""
import math

import torch
import torch.optim as optim
from pathlib import Path

from ..losses import rsad_loss
from ..metrics import compute_metrics
from ..utils.checkpoint import save_ckpt
from ..utils.logger import progress


def train_one_epoch(model, loader, opt, device, cfg):
    """一个 epoch 训练, 返回 avg_loss, avg_l_seg, avg_l_cls。"""
    model.extractor.eval()
    model.aligner.train()
    model.disc.train()
    tot_loss, tot_seg, tot_cls = 0.0, 0.0, 0.0
    n_samples = 0  # drop_last=True 时实际处理的样本数（分母）
    for imgs in progress(loader, desc="train"):
        imgs = imgs.to(device)
        pos_logit, neg_logit = model.train_forward(imgs)
        loss, l_seg, l_cls = rsad_loss(
            pos_logit, neg_logit,
            th_pos=cfg.th_pos if hasattr(cfg, "th_pos") else 0.5,
            th_neg=cfg.th_neg if hasattr(cfg, "th_neg") else -0.5,
            gamma=cfg.focal_gamma if hasattr(cfg, "focal_gamma") else 2.0,
            alpha=cfg.focal_alpha if hasattr(cfg, "focal_alpha") else 0.25,
        )
        opt.zero_grad()
        loss.backward()
        # 梯度裁剪: 抑制极端梯度（诊断显示 |g_disc| 可到 100+）;
        # 阈值 10.0 放宽, 避免压死正常学习信号（1.0 曾导致 connector 训练停滞）
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], max_norm=10.0)
        opt.step()
        n_samples += len(imgs)
        tot_loss += loss.item() * len(imgs)
        tot_seg += l_seg.item() * len(imgs)
        tot_cls += l_cls.item() * len(imgs)
    return tot_loss / n_samples, tot_seg / n_samples, tot_cls / n_samples


def train(cfg):
    """完整训练流程（从 cfg 驱动）。"""
    import torch
    from torch.utils.data import DataLoader
    from ..data.datasets import NormalTrainDataset, TestDataset, mask_collate
    from ..data.transforms import train_transform, test_transform
    from ..models.rsad_model import RSADModel
    from ..utils.seed import set_seed
    from ..utils.logger import Logger

    set_seed()
    device = torch.device(cfg.device if cfg.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    cls_filter = [cfg.cls] if getattr(cfg, "cls", None) else None
    out_dir = Path(cfg.out if hasattr(cfg, "out") else "checkpoints")
    if cls_filter:
        out_dir = out_dir / cfg.cls          # per-class: checkpoints/<class>/
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "train.log")

    # 数据
    root = Path(cfg.data_root)
    train_ds = NormalTrainDataset(root / "train", train_transform(cfg.crop), classes=cls_filter)
    test_ds = TestDataset(root / "test", test_transform(cfg.crop), classes=cls_filter)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,
                              num_workers=cfg.num_workers if hasattr(cfg, "num_workers") else 0,
                              drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch, shuffle=False,
                             num_workers=cfg.num_workers if hasattr(cfg, "num_workers") else 0,
                             collate_fn=mask_collate)
    logger.write(f"训练(正常): {len(train_ds)} 张 | 测试: {len(test_ds)} 张\n")

    # 解析 --noise-smooth "k5s1.0" -> {"kernel":5,"sigma":1.0}（须在 RSADModel 构建前;
    # yaml 直接写 dict 亦可）
    if isinstance(getattr(cfg, "noise_smooth", None), str):
        import re
        m = re.fullmatch(r"k(\d+)s([\d.]+)", cfg.noise_smooth.strip())
        cfg.noise_smooth = ({"kernel": int(m.group(1)), "sigma": float(m.group(2))}
                            if m else None)
    if getattr(cfg, "noise_smooth", None):
        logger.write(f"[noise_smooth] {cfg.noise_smooth}\n")

    # 模型
    model = RSADModel(cfg).to(device)

    # 优化器: Adam 双 lr 组（no_adapter 时 aligner 无参数, 跳过空组）
    param_groups = []
    if any(p.requires_grad for p in model.aligner.parameters()):
        param_groups.append({"params": [p for p in model.aligner.parameters() if p.requires_grad],
                             "lr": cfg.lr})
    if any(p.requires_grad for p in model.disc.parameters()):
        param_groups.append({"params": [p for p in model.disc.parameters() if p.requires_grad],
                             "lr": cfg.lr_d})
    opt = optim.Adam(param_groups, weight_decay=cfg.wd if hasattr(cfg, "wd") else 1e-5)

    # 断点续训
    start_epoch = 1
    best = 0.0
    if hasattr(cfg, "resume") and cfg.resume:
        from ..utils.checkpoint import load_ckpt
        saved_epoch, saved_best, _ = load_ckpt(model, cfg.resume, device)
        start_epoch = saved_epoch + 1
        best = saved_best
        logger.write(f"[resume] 从 {cfg.resume} 恢复: 起始 epoch={start_epoch}, best={best:.4f}\n")

    # cosine lr 调度: 双参数组共同衰减到 0, 抑制后期震荡
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, last_epoch=start_epoch - 2)

    for epoch in range(start_epoch, cfg.epochs + 1):
        avg_loss, avg_seg, avg_cls = train_one_epoch(model, train_loader, opt, device, cfg)

        # 周期性评估
        eval_every = cfg.eval.every_n_epochs if hasattr(cfg, 'eval') else 10
        if epoch % eval_every == 0 or epoch == 1:
            model.extractor.eval(); model.aligner.eval(); model.disc.eval()
            i_auc, p_auc = compute_metrics(model, test_loader, device)
            model.aligner.train(); model.disc.train()
            msg = (f"epoch {epoch:3d}/{cfg.epochs} | loss={avg_loss:.4f} "
                   f"(seg={avg_seg:.4f} cls={avg_cls:.4f}) "
                   f"| I-AUROC={i_auc:.4f} | P-AUROC={p_auc:.4f}")
            logger.write(msg + "\n")
            if not math.isnan(i_auc) and i_auc > best:
                best = i_auc
                save_ckpt(model, out_dir, "best.pt", epoch, best,
                          cfg.sigma if hasattr(cfg, "sigma") else 0.015)
                logger.write(f"  -> 保存 best.pt (I-AUROC={i_auc:.4f})\n")
        else:
            logger.write(f"epoch {epoch:3d}/{cfg.epochs} | loss={avg_loss:.4f} "
                         f"(seg={avg_seg:.4f} cls={avg_cls:.4f})\n")

        # 每轮保存 latest.pt
        save_ckpt(model, out_dir, "latest.pt", epoch, best,
                  cfg.sigma if hasattr(cfg, "sigma") else 0.015)
        scheduler.step()

    # 最终评估
    model.extractor.eval(); model.aligner.eval(); model.disc.eval()
    i_auc, p_auc = compute_metrics(model, test_loader, device)
    logger.write(f"\n最终: I-AUROC={i_auc:.4f} | P-AUROC={p_auc:.4f} "
                 f"| 最优模型在 {out_dir / 'best.pt'}\n")
    logger.close()
    return model
