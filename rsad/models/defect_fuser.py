"""Stage 3 — DefectFeatureFuser（论文 §3.3,式(6)）

fused = aligned + σ·N(0,1)。训练时对 aligned.detach() 调用，梯度不回传 A。

可选改进（noise_smooth）: 噪声先采样再空间高斯模糊, 生成连续异常区域,
匹配 hole/scratches 等区域型缺陷（诊断: i.i.d. 孤立噪点模式对 bracket_black
的连续缺陷无响应）。默认 None 保持论文原样。
"""
import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur


class DefectFeatureFuser(nn.Module):
    def __init__(self, sigma=0.015, noise_smooth=None):
        """noise_smooth: None = i.i.d. 噪声（论文原样）;
        dict {kernel: 奇数 int, sigma: float} = 高斯模糊噪声。
        """
        super().__init__()
        self.sigma = sigma
        self.noise_smooth = noise_smooth

    def forward(self, aligned):
        noise = torch.randn_like(aligned)
        if self.noise_smooth:
            k = self.noise_smooth.get("kernel", 5)
            s = self.noise_smooth.get("sigma", 1.0)
            noise = gaussian_blur(noise, kernel_size=k, sigma=s)
        return aligned + self.sigma * noise
