"""随机种子"""
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确定性卷积（代价是少量速度），保证实验可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
