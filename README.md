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

## 数据与权重获取

### 数据（MPDD 数据集，约 1.8GB）

数据集**不随代码仓库分发**，按下列任一途径获取后放到约定路径即可：

**途径一：官方 GitHub（推荐）**
- 仓库：https://github.com/stepanje/MPDD
- 页面找 "Download"/"Clone" 按钮，或直接 `git clone https://github.com/stepanje/MPDD`
- 该仓库即数据集本体（含 `train/`、`test/` 目录）

**途径二：本仓库种子文件**
- 仓库根目录的 `MPDD.torrent`（17.8KB）用 BitTorrent 客户端（qBittorrent/µTorrent）打开下载

**途径三：国内镜像**
- 超神经 Hyper.AI：https://orion.hyper.ai/datasets/31541

**v3 数据划分**：训练使用 `MPDD_split/train`，验证使用 `MPDD_split/val`，最终测试使用 `MPDD_split/test`。最终测试集只在训练结束后评估一次。

```bash
python tools/setup_data.py --path ../rsad/data/MPDD  # 检查原始数据
python tools/setup_data.py --clone    # 缺失时自动从官方仓库克隆到约定路径
```

**目录结构要求**（校验通过即正确）：
```
rsad/data/MPDD/
├── train/<类别>/good/*.png          # 训练正常样本（6 类共 888 张）
└── test/<类别>/good/…               # 测试正常
    test/<类别>/<缺陷类型>/*.png      # 缺陷样本 + 同名 *_mask.png 掩码
```

### 预训练权重（WRN50 ImageNet）

**方式一：自动下载（推荐）**
```bash
python tools/download_weights.py      # 默认下载到 ../rsad/weights/wrn50_imagenet.bin
```
- 下载源：HuggingFace 镜像 `hf-mirror.com`（国内可达性较好）

**方式二：手动下载**
- 官方权重：torchvision `wide_resnet50_2` 的 IMAGENET1K_V2 权重（`pytorch_model.bin`，约 180MB）
- 放置到 `../rsad/weights/wrn50_imagenet.bin`

**兜底**：若未提供权重，代码会自动尝试 torchvision 在线加载（需能访问 pytorch.org）。

**校验**：启动训练/评估时打印 `[backbone] 本地权重 ... 加载 320/320 键` 即为成功。

> 路径规则：`data_root`/`backbone_weights` 等相对路径统一按**项目根目录**（`rsad_v2` 所在目录）解析，因此从任意 cwd 运行结果一致。

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
    --out checkpoints_v3 --split val
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
