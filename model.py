"""
模型定义模块

构建基于 ResNet 的肺炎 CT 影像分类模型

设计思路：
1. 使用 ImageNet 预训练的 ResNet 作为骨干网络
2. 替换最后的分类头，改为二分类输出
3. 添加 Dropout 防止过拟合
4. 可选择性地冻结部分底层参数
"""
import torch
import torch.nn as nn
from torchvision import models
import config


class PneumoniaCTClassifier(nn.Module):
    """
    肺炎 CT 影像分类器

    架构设计理由：
    ┌─────────────────────────────────────────────────────────┐
    │  Input: 3×224×224 CT 图像                               │
    │       ↓                                                 │
    │  Conv1(64) + BN + ReLU + MaxPool                        │
    │       ↓                                                 │
    │  Stage1: 3× Bottleneck(256)  ← 可能冻结                 │
    │       ↓                                                 │
    │  Stage2: 4× Bottleneck(512)  ← 可能冻结                 │
    │       ↓                                                 │
    │  Stage3: 6× Bottleneck(1024) ← 参与训练                 │
    │       ↓                                                 │
    │  Stage4: 3× Bottleneck(2048) ← 参与训练                 │
    │       ↓                                                 │
    │  AdaptiveAvgPool2d → Flatten                            │
    │       ↓                                                 │
    │  Dropout(p=0.5)  ← 防止过拟合                           │
    │       ↓                                                 │
    │  Linear(2048 → 2)  ← 二分类输出                         │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.5, freeze_stages=2):
        super().__init__()

        # 1. 加载预训练 ResNet
        # pretrained=True 时加载 ImageNet 预训练权重
        # 为什么用 ResNet50 而不是 ResNet18/34？
        # - ResNet50 的 Bottleneck 结构更高效（1×1 conv 降维 → 3×3 conv → 1×1 conv 升维）
        # - 特征表达能力更强，对医学图像的细微特征更敏感
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        # 2. 替换分类头
        # 原始 ResNet50 输出 1000 类（ImageNet 类别数）
        # 改为输出 2 类（肺炎阳性/阴性）
        num_features = self.backbone.fc.in_features  # ResNet50 为 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout 层
            nn.Linear(num_features, num_classes)  # 新的分类层
        )

        # 3. 冻结指定 stage 的参数
        # 为什么要冻结？
        # - 底层特征（边缘、纹理）是通用的，ImageNet 预训练已学得很好
        # - 冻结后减少可训练参数量，降低过拟合风险
        # - 对小数据集（几百张）尤其重要
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

    def _freeze_stages(self, num_stages):
        """
        冻结 ResNet 的前 N 个 stage

        ResNet50 的 stage 结构：
        - Stage 0: conv1 (第一个卷积层)
        - Stage 1: layer1 (3 个 Bottleneck, 输出 256 通道)
        - Stage 2: layer2 (4 个 Bottleneck, 输出 512 通道)
        - Stage 3: layer3 (6 个 Bottleneck, 输出 1024 通道)
        - Stage 4: layer4 (3 个 Bottleneck, 输出 2048 通道)
        """
        # 冻结 conv1 和 bn1
        if num_stages >= 1:
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False

        # 冻结后续 layer
        layers = [
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]
        for i in range(min(num_stages, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False

        # 统计冻结参数比例
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"冻结参数比例: {frozen_params / total_params * 100:.1f}% "
              f"({frozen_params:,} / {total_params:,})")

    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


def build_model():
    """
    构建模型并打印信息

    Returns:
        model: 构建好的 PneumoniaCTClassifier
    """
    model = PneumoniaCTClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        dropout_rate=config.DROPOUT_RATE,
        freeze_stages=config.FREEZE_STAGES,
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型: {config.MODEL_NAME}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")
    print(f"输入尺寸: 3 × {config.IMG_SIZE} × {config.IMG_SIZE}")
    print(f"分类类别: {config.NUM_CLASSES} (阳性/阴性)\n")

    return model


def to(device):
    return None