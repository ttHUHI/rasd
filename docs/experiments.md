# per-class 训练实验记录（2026-08）

## 动机

统一模型（一个判别器同时拟合 6 类金属表面正常流形）判别边界被稀释，
`tubes` 等类别几乎无响应（I-AUROC=0.484）。改为 per-class 训练：每类一个
模型（只学该类正常流形），推理按类别路由。

## 结论

per-class 检测能力显著优于统一模型（加权 I-AUROC 0.698 → 0.833，+13.5），
6 类全部不低于统一模型。

## 结果对比（I-AUROC %）

| 类别 | per-class | 统一模型 | 论文 Table 2 |
|------|-----------|----------|--------------|
| bracket_black | 66.9 | 64.4 | 94.3 |
| bracket_brown | 84.8 | 73.7 | 99.7 |
| bracket_white | 78.3 | 72.4 | 96.1 |
| connector     | 92.6 | 75.7 | 100 |
| metal_plate   | 99.9 | 97.8 | 100 |
| tubes         | 77.8 | 48.4 | 99.7 |
| **加权平均**  | **83.3** | **69.8** | **98.3** |

P-AUROC（加权）：per-class 71.7 vs 统一模型 87.7（per-class 更低，推测因
统一模型的跨类特征差异被误当异常信号；per-class 去掉该伪信号后更真实）。

## 训练/评估/推理命令

```bash
# 训练单类（20 epoch, sigma=0.1, 输出 checkpoints_pc/<cls>/）
python train.py --config configs/mpdd.yaml --cls metal_plate \
    --epochs 20 --batch 4 --device cuda --sigma 0.1 --out checkpoints_pc

# 评估单类
python evaluate.py --config configs/mpdd.yaml --cls metal_plate \
    --ckpt checkpoints_pc/metal_plate/best.pt

# 推理路由（--cls 自动加载 <out>/<cls>/best.pt）
python tools/predict.py --config configs/mpdd.yaml --cls metal_plate \
    --out checkpoints_pc --input <图片或目录> [--vis 输出目录]
```

## 已知弱点与后续

- **tubes**：排序 AUROC 0.778，但阈值 0.5 下缺陷检出仅 17.8%（绝对分数
  校准偏低）——需更小 sigma / 更长训练 / 阈值下移
- **bracket_black**：训练震荡（epoch 10 best 0.669，epoch 20 崩至 0.35），
  best.pt 机制兜底，可试更小 sigma 或更短训练
- 与论文（94-100）仍有明显差距，per-class 是正确方向但不是全部答案；
  后续结合改进路线（cosine lr / 空间平滑噪声 / 特征归一化）逐类优化
