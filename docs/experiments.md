# per-class 训练实验记录（2026-08）

## 动机

统一模型（一个判别器同时拟合 6 类金属表面正常流形）判别边界被稀释，
`tubes` 等类别几乎无响应（I-AUROC=0.484）。改为 per-class 训练：每类一个
模型（只学该类正常流形），推理按类别路由。

## 结论

per-class 检测能力显著优于统一模型（加权 I-AUROC 0.698 → 0.870，+17.2），
6 类全部不低于统一模型。

## 训练稳定化改进（trainer.py）

- cosine lr 调度（双参数组衰减到 0）
- 梯度裁剪 clip_grad_norm_(10.0)。注意: 曾用 1.0 导致 connector 训练停滞
  （0.93→0.65 回归），放宽到 10.0 后恢复且 6 类全面提升。

## 结果对比（I-AUROC %，best.pt 评估）

| 类别 | 统一模型 | per-class 20ep | per-class 30ep 改进后 | 论文 Table 2 |
|------|----------|----------------|----------------------|--------------|
| bracket_black | 64.4 | 66.9 | **74.1** | 94.3 |
| bracket_brown | 73.7 | 84.8 | **87.8** | 99.7 |
| bracket_white | 72.4 | 78.3 | **80.8** | 96.1 |
| connector     | 75.7 | 92.6 | **93.8** | 100 |
| metal_plate   | 97.8 | 99.9 | **100.0** | 100 |
| tubes         | 48.4 | 77.8 | **84.6** | 99.7 |
| **加权平均**  | **69.8** | **83.3** | **87.0** | **98.3** |

P-AUROC（加权）：per-class 30ep ≈ 73.4（统一模型 87.7 含跨类伪信号，per-class
更真实）。

## 训练/评估/推理命令

```bash
# 训练单类（30 epoch, cosine lr + 梯度裁剪, 输出 checkpoints_pc30/<cls>/）
python train.py --config configs/mpdd.yaml --cls metal_plate \
    --epochs 30 --batch 4 --device cuda --sigma 0.1 --out checkpoints_pc30

# 评估单类
python evaluate.py --config configs/mpdd.yaml --cls metal_plate \
    --ckpt checkpoints_pc30/metal_plate/best.pt

# 推理路由（--cls 自动加载 <out>/<cls>/best.pt）
python tools/predict.py --config configs/mpdd.yaml --cls metal_plate \
    --out checkpoints_pc30 --input <图片或目录> [--vis 输出目录]
```

## 已知弱点与后续

- **tubes**：排序 AUROC 0.846，但阈值 0.5 下缺陷检出率仍偏低（绝对分数
  校准问题）——可阈值下移或继续小 sigma 训练
- **bracket_black**：best 在 ep20（0.741），ep30 略回落（0.697）——训练仍
  有波动，best.pt 机制兜底
- 与论文（94-100）仍有差距；下一步候选：空间平滑噪声（匹配真实缺陷的
  连续区域）、特征归一化 + 相对 sigma、逐类调 sigma
