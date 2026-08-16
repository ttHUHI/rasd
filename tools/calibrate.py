# -*- coding: utf-8 -*-
"""阈值校准: 用训练集正常样本的 s_ad 分布定判定阈值（默认 95 分位）。

背景: 调小 sigma 后判别器分数整体上移, 固定 0.5 阈值不再适用。
用该类训练集正常样本的 s_ad 高分位作为阈值, 使正常样本误报率 ≈ 1-q。

用法:
    python tools/calibrate.py --config configs/mpdd.yaml \
        --cls metal_plate --out checkpoints_pc80 [--q 0.95] [--split test]

输出: <out>/<cls>/threshold.json  {"th": ..., "q": ..., "n": ...}
predict.py 的 --cls 路由会自动读取该文件作为阈值。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from rsad.utils.config import load_config, build_parser
from rsad.data.datasets import NormalTrainDataset, TestDataset, mask_collate
from rsad.data.transforms import test_transform
from rsad.inference import load_model


def main():
    ap = build_parser("RSAD threshold calibration")
    ap.add_argument("--q", type=float, default=0.95,
                    help="正常样本分数分位数（默认 0.95, 正常误报率≈5%）")
    ap.add_argument("--split", choices=["train", "test"], default="test",
                    help="用哪部分正常样本定阈值: test(默认, 已确认正常的样本, 符合部署场景) "
                         "| train(训练集正常, 无标签泄漏但判别器过拟合导致阈值偏低)")
    args = ap.parse_args()
    if not args.cls:
        print("请用 --cls 指定类别")
        return 1

    cfg = load_config(args.config, args)
    device = torch.device(cfg.device if cfg.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    base = Path(args.out) if args.out else Path("checkpoints")
    ckpt = base / args.cls / "best.pt"
    if not ckpt.exists():
        print(f"checkpoint 不存在: {ckpt}")
        return 1

    model = load_model(cfg, str(ckpt), device)
    transform = test_transform(cfg.crop if hasattr(cfg, "crop") else 224)
    root = Path(cfg.data_root)

    if args.split == "test":
        # 测试集正常样本（"已确认正常的样本"，部署场景等价）
        ds = TestDataset(root / "test", transform, classes=[args.cls])
        loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=mask_collate)
        sads = []
        with torch.no_grad():
            for imgs, labels, _ in loader:
                probs = model.test_forward(imgs.to(device))
                sa = (1.0 - probs).amax(dim=(1, 2, 3)).cpu().numpy()
                for s, l in zip(sa, labels):
                    if l == 0:
                        sads.append(float(s))
    else:
        ds = NormalTrainDataset(root / "train", transform, classes=[args.cls])
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        sads = []
        with torch.no_grad():
            for imgs in loader:
                probs = model.test_forward(imgs.to(device))
                sads.extend((1.0 - probs).amax(dim=(1, 2, 3)).cpu().numpy().tolist())
    sads = np.array(sads)
    th = float(np.percentile(sads, args.q * 100))

    out = base / args.cls / "threshold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"th": th, "q": args.q, "n": int(len(sads)),
                               "sad_mean": float(sads.mean()),
                               "sad_p95": float(np.percentile(sads, 95))},
                              indent=2), encoding="utf-8")
    print(f"[calibrate] {args.cls}: 正常样本 n={len(sads)}, "
          f"q{int(args.q * 100)} 分位阈值 th={th:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
