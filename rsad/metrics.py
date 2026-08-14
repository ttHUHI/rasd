"""评价指标: I-AUROC (图像级) + P-AUROC (像素级)

图像级分数 = 异常图最大值（论文 §3.4: max over anomaly map, 异常图 = 1 - 正常概率）。
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def compute_metrics(model, loader, device):
    """运行一次测试集评估, 返回 (I-AUROC, P-AUROC)。

    model: RSADModel (eval mode)
    loader: DataLoader (mask_collate 返回 (imgs, labels, masks))
    """
    model.extractor.eval()
    model.aligner.eval()
    model.disc.eval()

    sads, labels = [], []
    flat_anom, flat_gt = [], []

    for imgs, lbl, masks in loader:
        imgs = imgs.to(device)
        probs = model.test_forward(imgs)             # (B,1,28,28) 正常概率
        # 图像级分数 = 异常图最大值（论文 §3.4: max over anomaly map）
        s_ad = (1.0 - probs).amax(dim=(1, 2, 3)).cpu().numpy()  # (B,) image score
        sads.append(s_ad)
        labels.append(np.asarray(lbl))
        # 像素级: 异常图 = 1 - prob
        anom = (1.0 - probs)
        anom = F.interpolate(anom, size=(224, 224), mode="bilinear",
                             align_corners=False).squeeze(1).cpu().numpy()
        for j, m in enumerate(masks):
            if m is not None:
                flat_anom.append(anom[j].ravel())
                flat_gt.append(m.squeeze(0).numpy().ravel())

    sads = np.concatenate(sads)
    labels = np.concatenate(labels)
    i_auroc = roc_auc_score(labels, sads) if len(set(labels)) > 1 else float("nan")
    p_auroc = float("nan")
    if flat_anom:
        an, gt = np.concatenate(flat_anom), np.concatenate(flat_gt)
        if len(set(gt)) > 1:
            p_auroc = roc_auc_score(gt, an)
    return i_auroc, p_auroc
