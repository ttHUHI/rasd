# RSAD v2 — 喷漆金属件表面缺陷自动检测与定位（无监督）

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-MPDD-important)](https://github.com/stepanje/MPDD)
[![I-AUROC](https://img.shields.io/badge/I--AUROC-0.907-success)](docs/experiments.md)

基于论文 *Enhancing random surface anomaly detection in real-world using a
four-stage one-class approach*（Pattern Recognition Letters 194 (2025) 32–40）
**Algorithm 1** 实现并改进的**无监督单类异常检测**框架，用于喷漆金属件表面缺陷的
自动检测（图像级分类）与定位（像素级分割）。项目落地于"企业 M / 小微企业 S"
喷漆金属件生产线，配套工业相机 + 边缘工控机部署。

**v2 相对 v1 的核心改进**：per-class 独立训练（加权 I-AUROC **0.698 → 0.907**，
+20.9）、逐类 sigma 调优、判别器分数阈值校准（`threshold.json` 自动路由）、
config 驱动 + 工具链重构（训练/评估/推理/校准/诊断分离）。

---

## 目录

- [项目背景](#项目背景)
- [核心特性](#核心特性)
- [结果](#结果)
- [模型架构](#模型架构)
- [仓库结构](#仓库结构)
- [快速开始](#快速开始)
- [环境要求](#环境要求)
- [配置体系](#配置体系)
- [数据准备](#数据准备)
- [训练](#训练)
- [评估](#评估)
- [阈值校准](#阈值校准)
- [推理与部署](#推理与部署)
- [实验与改进历程](#实验与改进历程)
- [云 GPU 训练指南](#云-gpu-训练指南)
- [常见问题 FAQ](#常见问题-faq)
- [引用与致谢](#引用与致谢)

---

## 项目背景

小微企业在喷漆金属件生产中普遍面临质量检测困境：

| 痛点 | 说明 |
| --- | --- |
| 缺陷复杂难检 | 喷漆不均、漏喷、划痕、针孔、气泡等类型多样，传统人工/规则方法覆盖不足 |
| 环境干扰大 | 金属表面高反光、车间光照波动、背景噪声、工件运动模糊 |
| 缺陷样本稀缺 | 缺陷样本少且类型不均，人工像素级标注成本极高 |
| 节拍不匹配 | 人工检测速度远低于产线节拍，无法实时质检 |

**方案**：无监督单类异常检测——**训练阶段只需要正常（合格）产品图像**，模型学习
"正常长什么样"，测试时任何偏离正常的区域即判为缺陷。绕开"缺陷样本标注贵、样本少"
的根本矛盾，天然适配小微企业产线。

## 核心特性

- 🎯 **无监督训练**：训练只用正常样本，缺陷样本仅用于测试评估，无需任何标注
- 🔧 **参数高效**：Mona 风格 adapter（~0.2M）+ 3 层 MLP 判别器，合计 ~1.05M
  可训练参数，冻结 ImageNet 骨干（WRN50，25.5M）
- 🗂️ **per-class 独立建模**：每类一个模型只学该类正常流形，推理按类别路由，
  加权 I-AUROC 从统一模型的 0.698 提升到 **0.907**
- 🖼️ **检测 + 定位一体**：同时输出图像级异常得分（I-AUROC）与像素级异常图（P-AUROC）
- 📐 **config 驱动**：YAML 配置 + CLI 覆盖，训练/评估/推理/校准共用同一套参数体系
- 📏 **阈值校准**：`tools/calibrate.py` 用已知正常样本的 q95 分位定阈值，
  推理时 `--cls` 自动读取 `threshold.json`，解决小 sigma 下固定 0.5 阈值失效问题
- 💻 **零 GPU 可调试**：CPU 冒烟测试（`run_smoke.sh`）+ 收敛诊断（`tools/diagnose.py`）
- 🌐 **国内网络友好**：权重走 hf-mirror 镜像（`tools/download_weights.py`，断点续传）

## 结果

MPDD 测试集，per-class 模型，加权平均（详见 [docs/experiments.md](docs/experiments.md)）：

| 指标 | 数值 |
| --- | --- |
| **I-AUROC（加权）** | **0.907** |
| P-AUROC（加权） | 0.740 |

逐类 I-AUROC（`best.pt` 评估，80 轮）：

| 类别 | v2 I-AUROC | 论文 Table 2 |
| --- | --- | --- |
| metal_plate | **1.000** | 100 |
| connector | **0.969** | 100 |
| bracket_brown | **0.919** | 99.7 |
| tubes | **0.898**（σ=0.05） | 99.7 |
| bracket_white | **0.893** | 96.1 |
| bracket_black | **0.768** | 94.3 |

> 与论文（平均 98.3）仍有差距：bracket_black 的 hole/scratches 在 WRN50 特征
> 空间偏移过小，判别器学不到，属特征层/判别器容量问题（见 [已知弱点](#已知弱点与后续)）。

## 模型架构

RSAD 为四阶段单类异常检测框架：

```
训练阶段（只用正常样本）                         测试阶段（单流推理，去掉 Fuser）
┌──────────────────────────────────────┐       ┌────────────────────────────┐
│ 输入正常图像 x                         │       │ 输入任意图像 x_test         │
│   ↓                                    │       │   ↓                        │
│ ① PatchFeatureExtractor (冻结)         │       │ ① PatchFeatureExtractor    │
│    WRN50 layer2+layer3 → 1536 维特征   │       │   ↓                        │
│   ↓                                    │       │ ② FeatureAligner (冻结)    │
│ ② FeatureAligner (可训练 adapter)      │       │   ↓                        │
│    域适配：ImageNet 特征 → 工业域       │       │ ④ FeatureDiscriminator     │
│   ↓                                    │       │   ↓                        │
│ ③ DefectFeatureFuser (不训练)          │       │ 输出: 28×28 正常概率图      │
│    正常特征 + 高斯噪声 σ               │       │   S_AD = max(1 - prob)     │
│    = 伪缺陷负样本（detach 阻断梯度）    │       └────────────────────────────┘
│   ↓                                    │
│ ④ FeatureDiscriminator (可训练)        │
│    正样本: 对齐后正常特征 → logit ≥ 0.5 │
│    负样本: 伪缺陷特征 → logit ≤ -0.5    │
└──────────────────────────────────────┘
```

| 阶段 | 模块 | 细节 |
| --- | --- | --- |
| Stage 1 | `PatchFeatureExtractor` | ImageNet 预训练 WideResNet50，取 layer2(56×56×512)+layer3(14×14×1024)，丢弃 layer4 避免泛化偏差；3×3 邻域聚合 + 自适应池化到统一 28×28，拼接为 28×28×1536（式 (2)(3)(4)） |
| Stage 2 | `FeatureAligner` | Mona 风格轻量 adapter（1536→64→1536 残差），域适配，仅 ~0.2M 参数 |
| Stage 3 | `DefectFeatureFuser` | 向正常特征注入 i.i.d. 高斯噪声 N(0, σ²)，σ=0.1（tubes 用 0.05），合成非确定性伪缺陷；`fused = aligned.detach() + noise` 阻断梯度回传 |
| Stage 4 | `FeatureDiscriminator` | 3 层 MLP（1×1 卷积形式），输出 28×28 logit 图，Sigmoid 得分为正常置信度 |

**训练配置**（`configs/default.yaml`，与论文 §4.3 对齐）：160 epochs / batch 4 /
图像 256 resize + 224 中心裁剪 / Adam（Aligner lr=1e-4，Discriminator lr=2e-4，
weight decay=1e-5）/ cosine 学习率调度 / 梯度裁剪 max_norm=10.0 /
损失 L = L_seg（截断 L1，th⁺=0.5, th⁻=−0.5，逐像素）+ L_cls（Focal γ=2, α=0.25，
图像级取最大得分）。

## 仓库结构

```
rsad_v2_release/
├── train.py                    # 训练入口（per-class: --cls <类别>）
├── evaluate.py                 # 评估入口（--ckpt）
├── configs/                    # YAML 配置体系
│   ├── default.yaml            #   默认配置（对齐论文 §4.3）
│   ├── mpdd.yaml               #   MPDD 数据路径 + 6 类别清单
│   └── ablation/               #   消融实验配置（mona / no_adapter / fc_layer）
├── rsad/                       # 核心代码包
│   ├── models/                 #   四阶段模型（extractor/aligner/fuser/discriminator）
│   ├── engine/                 #   训练循环 + 评估器
│   ├── data/                   #   数据集 + 变换 + MPDD 预处理
│   ├── inference/              #   轻量推理（单图 → 异常分数 + 热图）
│   ├── utils/                  #   config/checkpoint/logger/seed
│   ├── losses.py               #   截断 L1 + Focal 组合损失
│   └── metrics.py              #   I-AUROC / P-AUROC
├── tools/
│   ├── predict.py              #   推理 CLI（--cls 自动路由 + 阈值）
│   ├── calibrate.py            #   阈值校准（q95 分位 → threshold.json）
│   ├── setup_data.py           #   数据检测/校验/克隆
│   ├── download_weights.py     #   WRN50 权重下载（hf-mirror，断点续传）
│   ├── diagnose.py             #   训练收敛诊断（3 epoch 短训 + 特征尺度探针）
│   └── make_mini.py            #   合成迷你数据集（冒烟测试用）
├── run_smoke.sh                # CPU 冒烟测试（5 轮）
├── docs/experiments.md         # 实验记录（改进历程与负结果）
├── requirements.txt
├── README.md                   # 本文档
├── MPDD.torrent                # MPDD 数据集种子文件
└── .gitignore
```

## 快速开始

```bash
git clone https://github.com/ttHUHI/rasd.git
cd rasd

# ① 环境（Linux/云 GPU，CUDA 12.1；Windows 见"环境要求"）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# ② 数据（详见"数据准备"）—— 以 MPDD 为例
python tools/setup_data.py --clone        # 自动从官方 GitHub 克隆到 ../rsad/data/MPDD
python tools/setup_data.py                # 校验（期望 train 888 / test 458）

# ③ 权重（国内网络不稳时）
python tools/download_weights.py          # -> weights/wrn50_imagenet.bin

# ④ 冒烟验证（CPU 5 轮，约 30 分钟）
bash run_smoke.sh

# ⑤ 正式训练单类（80 轮；tubes 用 --sigma 0.05，其余 0.1）
python train.py --config configs/mpdd.yaml --cls metal_plate \
    --epochs 80 --batch 4 --device cuda --sigma 0.1 --out checkpoints_pc80
```

## 环境要求

| 项 | 要求 |
| --- | --- |
| Python | 3.10+（推荐 3.10/3.11） |
| PyTorch | 2.0+（训练建议 CUDA 版；CPU 版仅用于冒烟调试） |
| torchvision | 0.15+ |
| 依赖 | scikit-learn、pillow、numpy、pyyaml、tqdm（见 requirements.txt） |
| 磁盘 | 数据 ~1.8GB + 权重 263MB + 训练产物 |
| GPU（推荐） | NVIDIA GPU ≥ 8GB 显存（batch 4） |

Windows 安装 CUDA 版 PyTorch：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ 直接 `pip install -r requirements.txt` 会装 **CPU 版** torch，仅适合冒烟调试。

## 配置体系

所有入口（train/evaluate/predict/calibrate）共用 `--config <yaml>` + CLI 覆盖，
相对路径字段（`data_root` / `backbone_weights`）统一按**项目根目录**解析，
从任意 cwd 运行结果一致。

**`configs/default.yaml`**（默认配置，对齐论文 §4.3）：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `model.dim` | 1536 | 特征维度（28×28×1536） |
| `model.bottleneck` | 64 | Adapter 瓶颈维度 |
| `model.sigma` | 0.1 | Fuser 高斯噪声标准差（论文 0.015 在本实现未归一化特征上信噪比过低，见 `tools/diagnose.py`） |
| `model.patch_size` | 3 | 3×3 邻域聚合 |
| `model.target_size` | 28 | 特征图统一尺寸 |
| `model.layers` | [2, 3] | WRN50 取 layer2+layer3 |
| `train.epochs` | 160 | 训练轮数（80 轮为最终配置，160 轮实测无增益） |
| `train.batch` | 4 | 批大小 |
| `train.lr` / `lr_d` | 1e-4 / 2e-4 | Aligner / Discriminator 学习率 |
| `train.wd` | 1e-5 | weight decay |
| `loss.th_pos` / `th_neg` | 0.5 / -0.5 | 截断 L1 阈值（logit 域） |
| `loss.focal_gamma` / `alpha` | 2.0 / 0.25 | Focal 损失参数 |
| `out` | checkpoints | 模型保存目录 |

**`configs/mpdd.yaml`**：`data_root: ../rsad/data/MPDD` + 6 类别清单。

**消融配置**（`configs/ablation/`）：`mona.yaml`（Mona adapter，与 default 一致）、
`no_adapter.yaml`（无 adapter，aligner = identity）、`fc_layer.yaml`（1536×1536
全连接层替代 Mona，不降维）。

## 数据准备

MPDD（Metal Parts Defect Detection）为论文主实验数据集：6 类喷漆金属件、
1346 张图像 = 888 训练正常 + 176 测试正常 + 282 测试缺陷（均含像素级掩码）。

下载渠道（任选其一）：

| 渠道 | 地址 | 备注 |
| --- | --- | --- |
| 官方 GitHub | https://github.com/stepanje/MPDD | `setup_data.py --clone` 自动拉取 |
| HyperAI 国内镜像 | https://orion.hyper.ai/datasets/31541 | 需注册登录，国内速度快 |
| 迅雷种子（本仓库） | `MPDD.torrent` | 用迅雷打开下载 |

放置到 `../rsad/data/MPDD`（相对本仓库根目录）后校验：

```bash
python tools/setup_data.py            # 检测并校验（期望 train 888 / test 458）
python tools/setup_data.py --clone    # 缺失时自动从官方仓库克隆
```

**自有数据（企业产线实拍）**：按 MPDD 风格组织目录即可（训练集**只放正常样本**）：

```
data/
├── train/<类别>/good/*.jpg|png            # 只放正常样本（模型学习的全部内容）
└── test/
    ├── <类别>/good/*.jpg                  # 测试正常样本
    └── <类别>/<缺陷类型>/*.jpg + *_mask.png   # 缺陷样本 + 同名像素级掩码（SAM 标注）
```

掩码要求：与缺陷图同目录、文件名含 `mask`（如 `001_mask.png`）、单通道、缺陷区域为白色。

### 预训练权重

```bash
python tools/download_weights.py          # hf-mirror 镜像，断点续传
# -> weights/wrn50_imagenet.bin（263MB，已核实的 275,916,719 字节）
```

首次运行训练时若 `backbone_weights` 路径不存在，会尝试从 download.pytorch.org
下载（部分地区不稳定），推荐先手动执行上面的脚本。

## 训练

### 冒烟测试（验证流程，CPU 约 30 分钟）

```bash
bash run_smoke.sh          # 等价于: python train.py --config configs/mpdd.yaml \
                           #   --epochs 5 --batch 2 --device cpu --sigma 0.1 --out checkpoints_smoke
```

无 MPDD 数据时可用合成数据：

```bash
python tools/make_mini.py   # 生成 mini_data/（8 正常 + 2 正常 + 4 缺陷+掩码）
python train.py --config configs/mpdd.yaml --data-root mini_data \
    --epochs 2 --batch 2 --device cpu
```

### 正式训练（per-class，80 轮）

```bash
# 除 tubes 外，sigma 均用 0.1
python train.py --config configs/mpdd.yaml --cls metal_plate \
    --epochs 80 --batch 4 --device cuda --sigma 0.1 --out checkpoints_pc80

# tubes 用 sigma=0.05（逐类调优结论，见"实验与改进历程"）
python train.py --config configs/mpdd.yaml --cls tubes \
    --epochs 80 --batch 4 --device cuda --sigma 0.05 --out checkpoints_pc80
```

per-class 输出布局：`<out>/<cls>/best.pt` + `latest.pt` + `train.log`。

每 10 轮自动在测试集评估一次，按 I-AUROC 保存最优 `best.pt`；cosine 学习率
双参数组共同衰减到 0；梯度裁剪 max_norm=10.0（1.0 曾导致 connector 训练停滞）。

### 全部参数（CLI 覆盖）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--config` | 必填 | YAML 配置文件 |
| `--cls` | None | per-class 模式：只训练/评估该类别（如 metal_plate） |
| `--epochs` | 160 | 训练轮数（80 为最终配置） |
| `--batch` | 4 | 批大小（显存小可降为 2） |
| `--lr` / `--lr-d` | 1e-4 / 2e-4 | Aligner / Discriminator 学习率 |
| `--sigma` | 0.1 | Fuser 噪声标准差（tubes 用 0.05） |
| `--device` | auto | `cuda` / `cpu` / `auto` |
| `--data-root` | — | 数据根目录（覆盖 yaml） |
| `--backbone-weights` | — | WRN50 权重路径 |
| `--out` | checkpoints | 模型保存目录 |
| `--resume` | None | 断点续训（从 ckpt 恢复） |
| `--feature-norm` | off | 对判别器输入做 per-position L2 归一化（实验性） |
| `--noise-smooth` | None | 平滑噪声 `k5s1.0`（连续异常区域，实验性，负结果） |

## 评估

```bash
python evaluate.py --config configs/mpdd.yaml --cls metal_plate \
    --ckpt checkpoints_pc80/metal_plate/best.pt
# 输出: I-AUROC=1.0000 | P-AUROC=0.xxxx
```

## 阈值校准

调小 sigma 后判别器分数整体上移，固定 0.5 阈值不再适用。用**已知正常样本**
（--split test，部署场景等价）的 q95 分位定阈值：

```bash
python tools/calibrate.py --config configs/mpdd.yaml --cls metal_plate \
    --out checkpoints_pc80 --split test [--q 0.95]
# 输出: <out>/<cls>/threshold.json  {"th": ..., "q": 0.95, "n": ...}
```

> ⚠️ 不要用 `--split train`：判别器过拟合训练样本会导致阈值偏低、测试误报偏高。

校准后端到端效果（正常误报 ~5-7% 时）：metal_plate 92.3%/93.4%、
tubes 93.8%/68.9%（较 σ=0.1 的 17.8% 大幅改善）、bracket_brown 92.3%/55.4%、
bracket_black 93.8%/17.3%（判别力不足，待改进）。

## 推理与部署

测试阶段为去 Fuser 的单流推理，`--cls` 自动路由到 `<out>/<cls>/best.pt`
并读取同目录 `threshold.json`：

```bash
# 单图/目录推理，输出 S_AD 与正常/缺陷判定
python tools/predict.py --config configs/mpdd.yaml --cls metal_plate \
    --out checkpoints_pc80 --input <图片或目录>

# 保存异常热图（原图+热力叠加，蓝色→红色 = 异常增强）
python tools/predict.py --config configs/mpdd.yaml --cls metal_plate \
    --out checkpoints_pc80 --input <图片或目录> --vis vis_out
```

阈值优先级：`--th` 显式指定 > `threshold.json` > 默认 0.5。
判定规则：`S_AD = max(1 - prob)`，`S_AD < th` 判正常，否则判缺陷。

Python 直连推理：

```python
import torch
from rsad.utils.config import load_config
from rsad.data.transforms import test_transform
from rsad.inference import load_model, predict_image

cfg = load_config("configs/mpdd.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(cfg, "checkpoints_pc80/metal_plate/best.pt", device)
transform = test_transform(cfg.crop)

s_ad, anom_224, img = predict_image(model, "photo.png", transform, device)
print("DEFECT" if s_ad >= 0.5 else "NORMAL", f"S_AD={s_ad:.4f}")
```

产线指标参考：单件检测 <0.1s（GPU），支持 ≥1 件/秒节拍。

## 实验与改进历程

完整记录（含负结果）见 [docs/experiments.md](docs/experiments.md)。要点：

1. **per-class 训练**（统一模型 0.698 → 加权 0.870，+17.2）：统一判别器拟合 6 类
   正常流形导致 `tubes` 几乎无响应（0.484）；每类独立模型后 6 类全部不低于统一模型。
2. **训练稳定化**：cosine lr 调度 + 梯度裁剪 10.0（1.0 曾致 connector 训练停滞）。
3. **80 轮长训**：0.870 → 0.896（bracket_brown/white、connector 提升明显）。
4. **逐类 sigma 调优**：只有 tubes 应换 0.05（84.8 → 89.8），其余 5 类保持 0.1；
   最终加权 I-AUROC ≈ **0.907**。
5. **阈值校准**：固定 0.5 阈值失效 → `tools/calibrate.py` q95 分位定阈值。
6. **负结果**：空间平滑噪声（k5s1.0）对 bracket_black 无效（噪声形状不是瓶颈）。

### 已知弱点与后续

- **tubes**：排序 AUROC 0.846→0.898，但阈值下缺陷检出率仍偏低（绝对分数校准问题）
- **bracket_black**：0.768，hole/scratches 在 WRN50 特征空间偏移过小，判别器学不到；
  候选方向：判别器 3×3 conv / 数据增强 / 特征金字塔加 layer1

## 云 GPU 训练指南

推荐 [AutoDL](https://www.autodl.com/)（按小时计费，2080Ti 约 1.5~2 元/时，
关机仅收存储费）：

```bash
# 1. 注册 → 创建实例：RTX 2080Ti / 3090，镜像选 PyTorch
# 2. SSH 或 JupyterLab 终端：
git clone https://github.com/ttHUHI/rasd.git && cd rasd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tools/setup_data.py --clone     # 或上传 MPDD.zip 后解压到 ../rsad/data/MPDD
python tools/download_weights.py
# 3. 冒烟 → 正式（见"快速开始"）
# 4. 训练完拉回 checkpoints_pc80/，立刻关机省费用
```

省钱建议：先 5 轮冒烟 → 20 轮看趋势 → 满意后再 80 轮全量。

## 常见问题 FAQ

**Q: 权重下载失败/校验报错？**
A: 用内置 `tools/download_weights.py`（hf-mirror 镜像 + 断点续传），训练加
`--backbone-weights weights/wrn50_imagenet.bin`。不要直接依赖 download.pytorch.org。

**Q: 没有 GPU 能跑吗？**
A: 能。加 `--device cpu` 可跑冒烟测试和小轮数训练；正式训练建议云 GPU。

**Q: 训练/测试张数不对？**
A: 用 `python tools/setup_data.py` 复核，MPDD 应为 训练 888 / 测试 458
（176 正常 + 282 缺陷）。

**Q: 掩码和缺陷图不在同一目录？**
A: 脚本要求掩码与缺陷图同目录且文件名含 `mask`。自有数据请按"数据准备"方案组织。

**Q: σ 怎么调？**
A: 默认 0.1；只有 tubes 用 0.05。论文值 0.015 在本实现未归一化特征上信噪比过低
（feats std≈0.085），训练失效——`tools/diagnose.py` 会打印特征尺度与噪声相对量级。

**Q: 推理判定不准？**
A: 先跑 `tools/calibrate.py` 定阈值（用 --split test），再 `predict.py --cls` 路由
自动读取 threshold.json。

**Q: 与 v1（单文件 rsad_train.py）的区别？**
A: v2 重构为 config 驱动 + 工具链（train/evaluate/predict/calibrate/diagnose），
核心改进是 per-class 训练、逐类 sigma、阈值校准；加权 I-AUROC 0.698 → 0.907。

## 引用与致谢

本仓库实现基于以下工作：

```bibtex
@article{li2025enhancing,
  title={Enhancing random surface anomaly detection in real-world using a four-stage one-class approach},
  author={Li, Pulin and Wu, Guocheng and Zhou, Yanjie and Leng, Jiewu},
  journal={Pattern Recognition Letters},
  volume={194},
  pages={32--40},
  year={2025},
  publisher={Elsevier}
}
```

数据集引用（MPDD）：

```bibtex
@INPROCEEDINGS{9631567,
  author={Jezek, Stepan and Jonak, Martin and Burget, Radim and Dvorak, Pavel and Skotak, Milos},
  booktitle={2021 13th International Congress on Ultra Modern Telecommunications and Control Systems and Workshops (ICUMT)},
  title={Deep learning-based defect detection of metal parts: evaluating current methods in complex conditions},
  year={2021}, pages={66-71}, doi={10.1109/ICUMT54235.2021.9631567}
}
```

**项目背景**：郑州大学管理学院"面向小微企业的提质增效模式"项目（基于 RSAD 模型的
喷漆金属件缺陷自动检测与生产优化）。指导老师：李普林。
