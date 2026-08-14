"""损失函数（论文式(7)(8)(9), logit 域截断 L1 + 概率域 Focal）

truncated_l1: 在判别器输出 logit 域计算（论文式(7)）:
    正常样本 logit 推到 ≥ th_pos(+0.5)，伪缺陷样本 logit 压到 ≤ th_neg(-0.5)。
    达到阈值后 loss=0（截断），且阈值远离 0 → 初始 logit≈0 时梯度恒为 ±1，
    避免概率域实现中"初始即截断、seg 梯度为 0"的训练失效。

focal_bce: 图像级 Focal Loss（论文式(8)），对 sigmoid(logit) 后的概率计算。
rsad_loss: L = L_seg + L_cls
"""
import torch
import torch.nn.functional as F


def truncated_l1(logits, is_normal, th=0.5):
    """截断 L1 损失, logit 域（论文式(7)）。

    正常(logit) → 推到 ≥ th_pos(0.5)：loss = relu(th_pos - logit)
    伪缺陷(logit) → 压到 ≤ th_neg(-0.5)：loss = relu(logit - th_neg)
    达到阈值后 loss = 0（截断）。
    """
    if is_normal:
        return torch.relu(th - logits).mean()
    else:
        return torch.relu(logits - th).mean()


def focal_bce(probs, target, gamma=2.0, alpha=0.25):
    """Focal Loss（论文式(8)），图像级，概率域。

    正常图像 S_AD ≈ 1（目标 1），伪缺陷图像 S_AD ≈ 0（目标 0）。
    gamma=2 降低易分样本权重，alpha=0.25 平衡正负。
    """
    pt = torch.where(target > 0.5, probs, 1.0 - probs)
    ce = F.binary_cross_entropy(probs, target, reduction="none")
    return (alpha * (1.0 - pt) ** gamma * ce).mean()


def rsad_loss(pos_logit, neg_logit, th_pos=0.5, th_neg=-0.5, gamma=2.0, alpha=0.25):
    """总损失 L = L_seg + L_cls（论文式(9)）。

    pos_logit / neg_logit: 判别器对正常/伪缺陷特征的 logit 输出 (B,1,28,28)。
    seg 用 logit 域截断 L1；cls 用 sigmoid 后图像级最大分数的 Focal。
    """
    l_seg = truncated_l1(pos_logit, is_normal=True, th=th_pos) \
            + truncated_l1(neg_logit, is_normal=False, th=th_neg)
    pos_img = torch.sigmoid(pos_logit).amax(dim=(1, 2, 3), keepdim=True)
    neg_img = torch.sigmoid(neg_logit).amax(dim=(1, 2, 3), keepdim=True)
    l_cls = focal_bce(pos_img, torch.ones_like(pos_img), gamma, alpha) \
            + focal_bce(neg_img, torch.zeros_like(neg_img), gamma, alpha)
    return l_seg + l_cls, l_seg, l_cls
