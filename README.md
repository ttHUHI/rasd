# RSAD — Random Surface Anomaly Detection（MPDD）

四阶段无监督工业缺陷检测与定位（论文：*Enhancing random surface anomaly detection in real-world using a four-stage one-class approach*, Pattern Recognition Letters）。

- Stage 1 Patch Feature Extractor（WRN50 预训练，layer2+3 → 1536 维 28×28 特征）
- Stage 2 Feature Aligner（Mona adapter 域适配）
- Stage 3 Defect Feature Fuser（特征域高斯噪声合成负样本）
- Stage 4 Feature Discriminator（3 层 1×1 Conv MLP，logit 域截断 L1 + Focal 损失）

## 结果（MPDD 测试集，per-class 模型，加权平均）

| 指标 | 数值 |
|------|------|
| I-AUROC（加权）| **0.907** |
| P-AUROC（加权）| 0.740 |
| 逐类 I-AUROC | metal_plate 1.000 / connector 0.969 / bracket_brown 0.919 / tubes 0.898 / bracket_white 0.893 / bracket_black 0.768 |

## 环境

```bash
conda create -n rsad python=3.10
conda activate rsad
pip install -r requirements.txt
```

## 数据获取

数据集不随仓库分发（约 1.8GB），获取途径：

1. **官方 GitHub**：https://github.com/stepanje/MPDD
2. **本仓库种子文件**：`MPDD.torrent`（需 BitTorrent 客户端）
3. **中文镜像**：https://orion.hyper.ai/datasets/31541

放置到 `../rsad/data/MPDD`（相对本仓库根目录）后校验：

```bash
python tools/setup_data.py            # 检测并校验（期望 train 888 / test 458）
python tools/setup_data.py --clone    # 缺失时自动从官方仓库克隆
```

预训练权重：`python tools/download_weights.py`（WRN50 ImageNet，默认放 `../rsad/weights/wrn50_imagenet.bin`）。

## 使用

```bash
# 训练单类（80 轮，tubes 用 --sigma 0.05，其余 0.1）
python train.py --config configs/mpdd.yaml --cls metal_plate \
    --epochs 80 --batch 4 --device cuda --sigma 0.1 --out checkpoints_pc80

# 评估单类
python evaluate.py --config configs/mpdd.yaml --cls metal_plate \
    --ckpt checkpoints_pc80/metal_plate/best.pt

# 推理（--cls 自动路由到 <out>/<cls>/best.pt 并读取阈值）
python tools/predict.py --config configs/mpdd.yaml --cls metal_plate \
    --out checkpoints_pc80 --input <图片或目录> [--vis <输出目录>]

# 阈值校准（已知正常样本 q95 分位）
python tools/calibrate.py --config configs/mpdd.yaml --cls metal_plate \
    --out checkpoints_pc80 --split test
```

冒烟测试：`bash run_smoke.sh`（CPU 5 轮）；收敛诊断：`python tools/diagnose.py sigma=0.1 epochs=3`。

## 目录结构

```
rsad_v2/
├── train.py / evaluate.py          # 训练/评估入口
├── configs/                        # default/mpdd/ablation 配置
├── rsad/                           # 核心代码（data/models/engine/utils/inference）
├── tools/                          # predict/calibrate/diagnose/setup_data/download_weights
├── checkpoints_pc80/               # 最终模型（best.pt + threshold.json）
├── MPDD.torrent                    # 数据种子文件
└── docs/experiments.md             # 实验记录（改进历程与负结果）
```

## 说明

- 改进历程、实验数据与负结果分析见 `docs/experiments.md` 与《改进思路》《最终》报告
- 80 轮为最终训练轮数（160 轮经实测无增益，判别器 40–60 轮即饱和）
