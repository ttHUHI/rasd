#!/bin/bash
# RSAD v2 冒烟测试: CPU 5 轮 MPDD
cd "$(dirname "$0")"
python train.py \
    --config configs/mpdd.yaml \
    --epochs 5 \
    --batch 2 \
    --device cpu \
    --sigma 0.1 \
    --out checkpoints_smoke
echo "Smoke test done."
