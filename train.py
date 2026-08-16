"""RSAD v2 训练入口"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsad.utils.config import load_config, build_parser
from rsad.engine.trainer import train


def main():
    ap = build_parser("RSAD training")
    args = ap.parse_args()
    cfg = load_config(args.config, args)
    train(cfg)


if __name__ == "__main__":
    main()
