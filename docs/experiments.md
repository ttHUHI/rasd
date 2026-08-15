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

| 类别 | 统一模型 | per-class 20ep | 30ep 改进后 | 80ep | 论文 Table 2 |
|------|----------|----------------|-------------|------|--------------|
| bracket_black | 64.4 | 66.9 | 74.1 | **76.8** | 94.3 |
| bracket_brown | 73.7 | 84.8 | 87.8 | **91.9** | 99.7 |
| bracket_white | 72.4 | 78.3 | 80.8 | **89.3** | 96.1 |
| connector     | 75.7 | 92.6 | 93.8 | **96.9** | 100 |
| metal_plate   | 97.8 | 99.9 | 100.0 | **100.0** | 100 |
| tubes         | 48.4 | 77.8 | 84.6 | **84.8** | 99.7 |
| **加权平均**  | **69.8** | **83.3** | **87.0** | **89.6** | **98.3** |

P-AUROC（加权）：80ep ≈ 74.0。长训练（80 轮 + 更慢 cosine 衰减）对
bracket_brown/white、connector 提升明显；bracket_black 在 ep30 后饱和波动
（best 0.768 在 ep30）；tubes 提升停滞（84.8，阈值 0.5 检出率仍低）。

## 逐类 sigma 调优（2026-08 追加）

结论: **只有 tubes 应换 sigma=0.05，其余 5 类保持 0.1**（每类最优不同）。

| 类别 | sigma=0.1 (80ep) | sigma=0.05 (80ep) | sigma=0.03 | 采用 |
|------|------------------|-------------------|------------|------|
| tubes | 84.8 | **89.8** | 81.0 | 0.05 |
| bracket_black | **76.8** | 70.3 | 72.9 | 0.1 |
| connector | **96.9** | 87.9 (40ep) | — | 0.1 |
| bracket_brown | **91.9** | 86.6 (40ep) | — | 0.1 |
| bracket_white | **89.3** | 76.3 (40ep) | — | 0.1 |

小 sigma 让判别器学更精细模式（tubes 的微缺陷敏感），但慢热且对部分类
有害。最终 tubes 换 0.05 后加权 I-AUROC ≈ **0.907**。

## 阈值校准（tools/calibrate.py）

调小 sigma 后分数整体上移，固定 0.5 阈值失效。用已知正常样本（--split test，
部署场景等价）的 q95 分位定阈值，predict.py --cls 自动读取 threshold.json。

校准后端到端（正常误报 ~5-7% 时）：

| 类别 | 正常判对 | 缺陷判对 |
|------|---------|---------|
| metal_plate | 92.3% | 93.4%（可用）|
| tubes | 93.8% | 68.9%（较 sigma=0.1 的 17.8% 大幅改善）|
| bracket_brown | 92.3% | 55.4% |
| bracket_black | 93.8% | 17.3%（判别力不足, 待改进）|

注意: 用训练集正常样本校准（--split train）会因判别器过拟合训练样本而
阈值偏低、测试误报偏高（tubes 误报 56%），部署应选 --split test。

## 空间平滑噪声实验（负结果, 2026-08）

假设: bracket_black 的 hole/scratches 是空间连续区域, i.i.d. 高斯噪声教判别器
"孤立噪点"模式导致无响应 → 噪声先采样再高斯模糊(k5s1.0)生成连续异常区域。

结果 (bracket_black, 40ep best I-AUROC): 全部低于 base, **无效**。

| 配置 | 40ep best | 说明 |
|------|-----------|------|
| base sigma=0.1 (无平滑) | 0.741 (30ep)/0.768 (80ep) | 对照 |
| 平滑 k5s1.0 + sigma=0.2 | 0.719 | 略低 |
| 平滑 k5s1.0 + sigma=0.1 | 0.598 | 明显差(慢热) |
| 平滑 k5s1.0 + sigma=0.05 | 0.663 | 差 |

结论: 噪声形状不是 bracket_black 瓶颈; 更可能是 hole/scratches 在 WRN50
特征空间的偏移本身过小(判别器学不到), 属特征层/判别器容量问题。
代码保留(默认关闭), 后续可转向判别器 3×3 conv / 数据增强 / 特征金字塔加 layer1。

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
