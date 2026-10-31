"""Stage 2 — FeatureAligner（论文 §3.2,真 Mona adapter）

完整 Mona：ScaledLayerNorm → down-projection → 多尺度 DWConv(3×3/5×5/7×7) → 1×1 聚合
→ GeLU → up-projection → scale 零初始化残差。参数量 ~0.42M（论文式(10)）。

消融（论文 Table 5）：
  - adapter_type=fc: 1536×1536 全连接（2.36M 参数）
  - use_adapter=False: identity（无 adapter，0 参数）
"""
import torch
import torch.nn as nn


class ScaledLayerNorm(nn.Module):
    """LayerNorm + 可学习 scale（Mona 原文"regulate distribution of input features"）。"""
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x: (B, C, H, W) → permute → LN → permute back
        b, c, h, w = x.shape
        out = x.permute(0, 2, 3, 1).reshape(-1, c)
        out = self.ln(out) * self.scale
        return out.reshape(b, h, w, c).permute(0, 3, 1, 2)


class MonaAdapter(nn.Module):
    """单分支 Mona adapter——多尺度深度可分离卷积版。

    论文 L94 描述：
      scaled LN → down → DWConv 3×3/5×5/7×7 (averaged) → 1×1 conv → GeLU → up
    三路 DWConv 的 groups=bottleneck（每通道独立卷积），参数量极低。
    """
    def __init__(self, dim=1536, bottleneck=64):
        super().__init__()
        self.norm = ScaledLayerNorm(dim)
        self.down = nn.Conv2d(dim, bottleneck, 1)          # 1536 → 64
        # 三路 DWConv，groups=bottleneck（每通道一个独立卷积核）
        self.dw3 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck)
        self.dw5 = nn.Conv2d(bottleneck, bottleneck, 5, padding=2, groups=bottleneck)
        self.dw7 = nn.Conv2d(bottleneck, bottleneck, 7, padding=3, groups=bottleneck)
        self.aggregate = nn.Conv2d(bottleneck, bottleneck, 1)  # 1×1 通道聚合
        self.act = nn.GELU()
        self.up = nn.Conv2d(bottleneck, dim, 1)            # 64 → 1536

    def forward(self, x):
        h = self.norm(x)
        h = self.down(h)                                    # (B,64,28,28)
        h = (self.dw3(h) + self.dw5(h) + self.dw7(h)) / 3.0  # 多尺度均值
        h = self.aggregate(h)
        h = self.act(h)
        return self.up(h)                                   # (B,1536,28,28)


class FCAdapter(nn.Module):
    """全连接 adapter（论文 Table 5 的 FC Layer 变体）。

    对每个空间位置做 1536×1536 线性映射（1×1 Conv 等价逐点全连接），
    参数量 = 1536² = 2.36M，对应论文"FC Layer 增加 2.36M 参数"。
    """
    def __init__(self, dim=1536):
        super().__init__()
        self.fc = nn.Conv2d(dim, dim, 1, bias=False)        # 1536×1536

    def forward(self, x):
        return self.fc(x)                                   # (B,1536,28,28)


class FeatureAligner(nn.Module):
    """Stage 2 域适配器——Mona adapter 残差包装。

    out = x + scale * adapter(x)，scale 零初始化 → 起步恒等, 稳定训练。
    adapter_type: "mona"（默认）| "fc"；use_adapter=False 时退化为恒等。
    """
    def __init__(self, dim=1536, bottleneck=64, adapter_type="mona", use_adapter=True):
        super().__init__()
        self.use_adapter = use_adapter
        if not use_adapter:
            self.adapter = nn.Identity()
        elif adapter_type == "fc":
            self.adapter = FCAdapter(dim)
        elif adapter_type == "mona":
            self.adapter = MonaAdapter(dim, bottleneck)
        else:
            raise ValueError(f"未知 adapter_type: {adapter_type}（可选 mona/fc）")
        if use_adapter:
            self.scale = nn.Parameter(torch.zeros(1))  # 论文: γ 零初始化

    def forward(self, x):
        if not self.use_adapter:
            return x
        return x + self.scale * self.adapter(x)
