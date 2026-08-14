"""Stage 3 — DefectFeatureFuser（论文 §3.3,式(6)）

fused = aligned + σ·N(0,1)。训练时对 aligned.detach() 调用，梯度不回传 A。
"""
import torch
import torch.nn as nn


class DefectFeatureFuser(nn.Module):
    def __init__(self, sigma=0.015):
        super().__init__()
        self.sigma = sigma

    def forward(self, aligned):
        noise = torch.randn_like(aligned) * self.sigma
        return aligned + noise
