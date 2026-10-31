"""RSAD v2 评估入口"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsad.utils.config import load_config, build_parser
from rsad.engine.evaluator import evaluate


def main():
    ap = build_parser("RSAD evaluation")
    args = ap.parse_args()
    cfg = load_config(args.config, args)
    if not hasattr(cfg, "ckpt") or not cfg.ckpt:
        print("请用 --ckpt 指定 checkpoint 路径")
        return
    evaluate(cfg)


if __name__ == "__main__":
    main()
