"""
数据集加载与预处理模块

包含:
1. CTImageDataset: 自定义数据集类，负责读取图像和标签
2. get_dataloaders: 创建训练/验证/测试数据加载器
3. 数据预处理和数据增强策略
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import config


class CTImageDataset(Dataset):
    """
    自定义 CT 图像数据集类

    为什么要自定义 Dataset？
    - PyTorch 的 ImageFolder 要求严格的目录结构（每个类别一个文件夹）
    - COVID-CT 数据集通常用 txt 文件标注每张图片的类别（0/1）
    - 自定义 Dataset 可以灵活处理各种数据组织方式

    txt 文件格式示例（每行一条记录）:
        data_path/to/image1.png,0
        data_path/to/image2.png,1
    """

    def __init__(self, txt_path=None, file_list=None, transform=None):
        """
        Args:
            txt_path: 包含文件路径和标签的 txt 文件
            file_list: 直接传入 (image_path, label) 的列表（替代 txt_path）
            transform: 图像预处理/增强操作
        """
        self.transform = transform
        self.samples = []

        if file_list is not None:
            # 直接传入数据列表
            self.samples = file_list
        elif txt_path and os.path.exists(txt_path):
            # 从 txt 文件读取
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 支持逗号或空格分隔
                    parts = line.replace(",", " ").split()
                    if len(parts) >= 2:
                        img_path = parts[0]
                        label = int(parts[1])
                        self.samples.append((img_path, label))
        else:
            raise FileNotFoundError(
                f"数据文件不存在: {txt_path}\n"
                "请检查 config.py 中的 DATA_DIR 和 txt 文件路径是否正确。"
            )

        print(f"加载 {len(self.samples)} 张图像 (路径: {txt_path or 'file_list'})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 读取图像
        # CT 图像可能是灰度图或 RGB 图，统一转为 3 通道
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transform():
    """
    训练集数据增强策略

    为什么要做数据增强？
    1. 医学图像数据集通常较小（几百到几千张），容易过拟合
    2. 数据增强相当于"创造"新样本，让模型看到更多变化
    3. 提高模型泛化能力，使其对图像的微小变化不敏感

    选择的增强操作：
    - RandomResizedCrop: 随机裁剪缩放，模拟不同距离/角度的拍摄
    - RandomHorizontalFlip: 水平翻转，CT 图像左右对称不影响诊断
    - RandomRotation: 随机旋转，模拟患者体位差异
    - ColorJitter: 轻微颜色抖动，增加鲁棒性
    - Normalize: ImageNet 归一化（与预训练权重一致）
    """
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE + 32, config.IMG_SIZE + 32)),  # 先放大再裁剪
        transforms.RandomResizedCrop(
            config.IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
        transforms.RandomRotation(degrees=15),   # 正负 15 度随机旋转
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 轻微颜色变化
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ])


def get_val_transform():
    """
    验证集/测试集预处理

    注意：验证/测试时不做随机增强，只做强确定性处理
    原因：需要保证评估结果的可复现性和一致性
    """
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ])


def get_dataloaders():
    """
    创建训练/验证/测试数据加载器

    返回值:
        train_loader, val_loader, test_loader, class_weights

    为什么需要 class_weights？
    - 计算各类别样本数量，用于加权交叉熵损失
    - 解决类别不平衡问题
    - 权重公式: weight_c = total_samples / (num_classes * count_c)
    """
    # 1. 创建数据集（先不带 transform）
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    # 尝试从 txt 文件加载数据
    train_dataset = None
    val_dataset = None
    test_dataset = None

    # 尝试加载训练集
    if os.path.exists(config.TRAIN_TXT):
        train_samples = _load_samples_from_txt(config.TRAIN_TXT)
        if config.USE_AUTO_SPLIT and len(train_samples) > 0:
            # 自动从训练集中划分验证集
            train_size = int(0.8 * len(train_samples))
            val_size = len(train_samples) - train_size
            # 先创建无 transform 的数据集用于划分
            base_dataset = CTImageDataset(file_list=train_samples, transform=None)
            train_dataset, val_dataset = random_split(
                base_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(config.SEED)
            )
            # 再分别应用不同的 transform
            train_dataset = _TransformSubset(train_dataset, train_transform)
            val_dataset = _TransformSubset(val_dataset, val_transform)
            print(f"自动划分: 训练集 {train_size} 张, 验证集 {val_size} 张")
        else:
            train_dataset = CTImageDataset(
                file_list=train_samples, transform=train_transform
            )

    # 尝试加载独立验证集
    if val_dataset is None and os.path.exists(config.VAL_TXT):
        val_samples = _load_samples_from_txt(config.VAL_TXT)
        if val_samples:
            val_dataset = CTImageDataset(
                file_list=val_samples, transform=val_transform
            )

    # 尝试加载测试集
    if os.path.exists(config.TEST_TXT):
        test_samples = _load_samples_from_txt(config.TEST_TXT)
        if test_samples:
            test_dataset = CTImageDataset(
                file_list=test_samples, transform=val_transform
            )

    if train_dataset is None:
        raise FileNotFoundError(
            "未找到任何数据！请检查:\n"
            f"1. config.py 中的 DATA_DIR 是否正确\n"
            f"2. 训练集 txt 文件是否存在: {config.TRAIN_TXT}\n"
            f"3. 或使用 file_list 方式直接传入数据"
        )

    # 计算类别权重（用于加权交叉熵）
    class_weights = _compute_class_weights(train_dataset)

    # 2. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,  # 加速 GPU 数据传输
        drop_last=True    # 丢弃最后一个不完整的 batch
    )

    val_loader = DataLoader(
        val_dataset if val_dataset else train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader, class_weights


def _load_samples_from_txt(txt_path):
    """从 txt 文件加载样本列表"""
    samples = []
    if not os.path.exists(txt_path):
        return samples
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                img_path = parts[0]
                label = int(parts[1])
                if os.path.exists(img_path):
                    samples.append((img_path, label))
    return samples


class _TransformSubset(Dataset):
    """为 Subset 包装 transform，直接从文件读取原始图像"""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        # 直接访问原始数据集的 samples 列表
        self.samples = subset.dataset.samples

    def __len__(self):
        return len(self.subset.indices)

    def __getitem__(self, idx):
        actual_idx = self.subset.indices[idx]
        img_path, label = self.samples[actual_idx]
        # 每次都从文件读取原始图像并应用 transform
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _wrap_with_transform(subset, transform):
    """为 Subset 包装 transform"""
    return _TransformSubset(subset, transform)


def _compute_class_weights(dataset):
    """
    计算类别权重

    原理：
    假设阳性 349 张，阴性 397 张，总数 746
    - 阳性权重 = 746 / (2 * 349) = 1.069
    - 阴性权重 = 746 / (2 * 397) = 0.939
    这样各类别在损失函数中的贡献趋于均衡
    """
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)

    labels = np.array(labels)
    num_classes = config.NUM_CLASSES
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)

    # 计算权重
    class_weights = total / (num_classes * class_counts + 1e-8)
    class_weights = torch.FloatTensor(class_weights)

    print(f"类别分布: {dict(zip(range(num_classes), class_counts.tolist()))}")
    print(f"类别权重: {class_weights.tolist()}")

    return class_weights
