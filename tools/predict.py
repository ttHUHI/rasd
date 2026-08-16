# -*- coding: utf-8 -*-
"""推理 CLI — 复用 train.py 的 --config/--ckpt/--input/--vis 参数。

用法:
    python tools/predict.py --config configs/mpdd.yaml \
        --cls metal_plate --input <图片或目录> [--vis <输出目录>]
        # 类别路由: 自动加载 checkpoints/metal_plate/best.pt
    python tools/predict.py --config configs/mpdd.yaml \
        --ckpt checkpoints/best.pt --input <图片或目录> [--vis <输出目录>]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from rsad.utils.config import load_config, build_parser
from rsad.data.transforms import test_transform
from rsad.inference import load_model, predict_image, save_heatmap, collect_images


def main():
    ap = build_parser("RSAD inference")
    args = ap.parse_args()
    # 类别路由: 给 --cls 且未给 --ckpt 时, 自动用 <out>/<cls>/best.pt
    if not args.ckpt:
        if args.cls:
            base = Path(args.out) if args.out else Path("checkpoints")
            args.ckpt = str(base / args.cls / "best.pt")
        else:
            print("请用 --ckpt 指定 checkpoint, 或 --cls 自动路由到 checkpoints/<cls>/best.pt")
            return 1
    if not Path(args.ckpt).exists():
        print(f"checkpoint 不存在: {args.ckpt}")
        print(f"提示: per-class 训练输出在 --out <目录> 下时, 推理需加 --out <目录> 保持路由一致")
        return 1
    if not args.input:
        print("请用 --input 指定图片或目录")
        return 1
    print(f"[路由] cls={args.cls or '(全类)'} ckpt={args.ckpt}")

    # 阈值: --th 显式指定 > 路由目录下的 threshold.json > 默认 0.5
    th = args.th
    if th is None:
        tfile = Path(args.ckpt).parent / "threshold.json"
        if tfile.exists():
            th = json.load(open(tfile, encoding="utf-8"))["th"]
    if th is None:
        th = 0.5
    print(f"[阈值] th={th:.4f}")

    cfg = load_config(args.config, args)
    device = torch.device(cfg.device if cfg.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(cfg, args.ckpt, device)
    transform = test_transform(cfg.crop if hasattr(cfg, "crop") else 224)

    files = collect_images(args.input)
    if not files:
        print(f"输入中没有图片: {args.input}")
        return 1

    vis_dir = Path(args.vis) if args.vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"推理 {len(files)} 张图片 (ckpt={args.ckpt}, device={device})")
    for i, f in enumerate(files, 1):
        s_ad, anom, img = predict_image(model, f, transform, device)
        tag = "NORMAL" if s_ad < th else "DEFECT"
        print(f"  [{i:3d}/{len(files)}] {f}  S_AD={s_ad:.4f}  {tag}")
        if vis_dir:
            save_heatmap(img, anom, vis_dir / f"{i:03d}_{f.stem}_heatmap.png", s_ad)

    if vis_dir:
        print(f"热图已保存到: {vis_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
