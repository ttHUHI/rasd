"""Stage 1 — PatchFeatureExtractor（论文 §3.1,式(1)-(4)）

WRN50 layer2+layer3 → 3×3 邻域 avgpool → bilinear resize → 通道拼接 → 1536 维 Normalcy Library.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PatchFeatureExtractor(nn.Module):
    def __init__(self, weights_path=None, layers=(2, 3), patch_size=3, target_size=28):
        super().__init__()
        self.layers = layers
        self.target_size = target_size

        # ── 加载 WRN50 backbone ──
        if weights_path:
            base = torchvision.models.wide_resnet50_2(weights=None)
            sd = torch.load(weights_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            sd = {k[7:] if (k.startswith("module.") or k.startswith("resnet.")) else k: v
                  for k, v in sd.items()}
            need = set(base.state_dict().keys())
            sd = {k: v for k, v in sd.items() if k in need}
            missing, unexpected = base.load_state_dict(sd, strict=False)
            print(f"[backbone] 本地权重 {weights_path}: 加载 {len(sd)}/{len(need)} 键, "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
        else:
            try:
                w = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2
            except AttributeError:
                w = "IMAGENET1K_V2"
            base = torchvision.models.wide_resnet50_2(weights=w)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2  # 28×28×512
        self.layer3 = base.layer3  # 14×14×1024

        # 冻结全部 backbone
        for p in self.parameters():
            p.requires_grad = False

        # ── 3×3 邻域聚合（论文式(2)(3)，neighborhood patch size p=3）──
        self.neighbor_pool = nn.AvgPool2d(kernel_size=patch_size, stride=1, padding=patch_size // 2)

    def forward(self, x):
        x = self.conv1(x)           # → 112×112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # → 56×56
        x = self.layer1(x)          # → 56×56
        f2 = self.layer2(x)         # → 28×28×512
        f3 = self.layer3(f2)        # → 14×14×1024

        # 3×3 邻域聚合
        f2 = self.neighbor_pool(f2)  # 28×28
        f3 = self.neighbor_pool(f3)  # 14×14

        # 统一到 target_size（28×28），论文式(4)"linear resize"
        if f3.shape[2] != self.target_size:
            f3 = F.interpolate(f3, size=(self.target_size, self.target_size),
                               mode="bilinear", align_corners=False)

        return torch.cat([f2, f3], dim=1)  # (B, 1536, 28, 28)
