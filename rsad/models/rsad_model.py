"""RSAD 四阶段模型组装。

train_forward: 返回 (pos_logit, neg_logit) 用于损失计算
test_forward:  单流推理, 返回 prob (B,1,28,28) 正常概率
"""
import torch
import torch.nn as nn

from .feature_extractor import PatchFeatureExtractor
from .feature_aligner import FeatureAligner
from .defect_fuser import DefectFeatureFuser
from .discriminator import FeatureDiscriminator


class RSADModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.extractor = PatchFeatureExtractor(
            weights_path=cfg.backbone_weights if hasattr(cfg, "backbone_weights") else None,
            layers=cfg.layers if hasattr(cfg, "layers") else (2, 3),
            patch_size=cfg.patch_size if hasattr(cfg, "patch_size") else 3,
            target_size=cfg.target_size if hasattr(cfg, "target_size") else 28,
        )
        self.aligner = FeatureAligner(
            dim=cfg.dim if hasattr(cfg, "dim") else 1536,
            bottleneck=cfg.bottleneck if hasattr(cfg, "bottleneck") else 64,
            adapter_type=cfg.adapter_type if hasattr(cfg, "adapter_type") else "mona",
            use_adapter=cfg.use_adapter if hasattr(cfg, "use_adapter") else True,
        )
        self.fuser = DefectFeatureFuser(
            sigma=cfg.sigma if hasattr(cfg, "sigma") else 0.015,
            noise_smooth=getattr(cfg, "noise_smooth", None),
        )
        self.disc = FeatureDiscriminator(
            in_ch=cfg.dim if hasattr(cfg, "dim") else 1536,
        )

        # 打印可训练参数量
        trainable = list(self.aligner.parameters()) + list(self.disc.parameters())
        n = sum(p.numel() for p in trainable)
        print(f"可训练参数: {n / 1e6:.2f} M（backbone 冻结）")

    def _normalize(self, feats):
        """per-position L2 归一化（可选，cfg.feature_norm=True 时启用）。

        每个 28×28 位置的 1536 维特征向量归一化到单位范数，使 sigma 相对
        单位特征有清晰语义，判别器输入尺度规整（默认关闭，保持论文原样）。
        """
        return feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-6)

    def train_forward(self, imgs):
        """训练前向: 返回正/负 logit。"""
        with torch.no_grad():
            feats = self.extractor(imgs)
        aligned = self.aligner(feats)               # 正样本
        if getattr(self.cfg, "feature_norm", False):
            aligned = self._normalize(aligned)
        fused = self.fuser(aligned.detach())        # 负样本——detach 阻断梯度传回 A
        pos_logit = self.disc(aligned)             # (B,1,28,28) logit
        neg_logit = self.disc(fused)               # (B,1,28,28)
        return pos_logit, neg_logit

    @torch.no_grad()
    def test_forward(self, imgs):
        """测试前向: 单流推理, 返回正常概率。"""
        feats = self.extractor(imgs)
        aligned = self.aligner(feats)
        if getattr(self.cfg, "feature_norm", False):
            aligned = self._normalize(aligned)
        return self.disc.prob(aligned)             # (B,1,28,28) 概率(Sigmoid 已含)
