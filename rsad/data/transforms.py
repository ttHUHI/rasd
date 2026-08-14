"""图像预处理: 256 resize + 224 中心裁剪 + ImageNet 归一化（论文 §4.3）"""
from torchvision import transforms as T

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def train_transform(crop=224):
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(crop),
        T.ToTensor(),
        T.Normalize(IMG_MEAN, IMG_STD),
    ])


def test_transform(crop=224):
    # 与训练相同
    return train_transform(crop)
