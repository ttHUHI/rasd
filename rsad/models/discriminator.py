"""Stage 4 — FeatureDiscriminator（论文 §3.4,图 1(c)）

3 层 1×1 Conv MLP(1536→512→128→1)，输出 logit（无 Sigmoid）。
训练时 loss 在 logit 域计算（论文式(7) 阈值 ±0.5）；推理时 sigmoid 得正常概率。
"""
import torch
import torch.nn as nn


class FeatureDiscriminator(nn.Module):
    def __init__(self, in_ch=1536, hidden=(512, 128)):
        super().__init__()
        self.fc1 = nn.Conv2d(in_ch, hidden[0], 1)
        self.fc2 = nn.Conv2d(hidden[0], hidden[1], 1)
        self.fc3 = nn.Conv2d(hidden[1], 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """x: (B, 1536, 28, 28) → logits: (B, 1, 28, 28)（未过 Sigmoid）"""
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

    def prob(self, x):
        """x: (B, 1536, 28, 28) → probs: (B, 1, 28, 28) 正常概率 (0~1)"""
        return torch.sigmoid(self.forward(x))
