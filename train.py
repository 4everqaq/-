"""
训练主程序

功能：
1. 加载数据集和模型
2. 配置优化器、学习率调度器、损失函数
3. 执行训练循环（训练 → 验证 → 早停检查 → 保存模型）
4. 输出训练日志和可视化结果

训练流程：
每个 Epoch 的步骤：
┌──────────────────────────────────────────────────┐
│  训练阶段 (Train):                                │
│  1. 模型设为 train 模式（启用 Dropout/BN 更新）    │
│  2. 按 batch 读取数据 → 前向传播 → 计算损失        │
│  3. 反向传播计算梯度 → 优化器更新权重              │
│  4. 累计平均 loss 和 accuracy                     │
│                    ↓                              │
│  验证阶段 (Validation):                            │
│  1. 模型设为 eval 模式（关闭 Dropout）              │
│  2. 不计算梯度（torch.no_grad）                    │
│  3. 按 batch 读取数据 → 前向传播 → 计算损失        │
│  4. 累计平均 loss 和 accuracy                     │
│                    ↓                              │
│  早停检查:                                        │
│  如果验证 loss 连续 patience 轮不下降 → 停止训练     │
│                    ↓                              │
│  保存模型:                                        │
│  如果当前验证 loss 是历史最低 → 保存权重            │
└──────────────────────────────────────────────────┘
"""
import matplotlib.pyplot as plt
plt.switch_backend("Agg")
import os
import time
import warnings

import torch

warnings.filterwarnings("ignore", category=UserWarning)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


import config
import utils
from model import build_model
from dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    执行一个 epoch 的训练

    Args:
        model: 神经网络模型
        loader: 训练数据 DataLoader
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备 (cuda/cpu)

    Returns:
        avg_loss: 平均训练损失
        avg_acc: 平均训练准确率
    """
    model.train()  # 切换到训练模式（启用 Dropout, BN 更新统计量）

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # 数据移到 GPU/CPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # === 前向传播 ===
        outputs = model(images)       # 模型预测
        loss = criterion(outputs, labels)  # 计算损失

        # === 反向传播 + 优化 ===
        optimizer.zero_grad()  # 清空上一轮的梯度（PyTorch 默认会累加梯度）
        loss.backward()        # 反向传播，计算当前 batch 的梯度
        optimizer.step()       # 用梯度更新模型权重

        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)  # 取概率最大的类别
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    执行验证（不计算梯度，速度更快，内存占用更小）

    @torch.no_grad() 装饰器的作用：
    - 告诉 PyTorch 不需要跟踪这个函数的计算图
    - 减少显存占用（不保存中间结果用于反向传播）
    - 加快推理速度（约 30-50%）

    Returns:
        avg_loss: 平均验证损失
        avg_acc: 平均验证准确率
    """
    model.eval()  # 切换到评估模式（关闭 Dropout, BN 使用运行统计量）

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc


def train():
    """主训练流程"""
    # ====== 1. 初始化 ======
    # 设置随机种子
    utils.set_seed(config.SEED)

    # 创建结果目录
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.FIGURE_DIR, exist_ok=True)

    # 设置日志
    log_file = os.path.join(config.LOG_DIR, "training.log")
    logger = utils.setup_logger(log_file=log_file)

    # 设备选择
    if config.DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.DEVICE)
    logger.info(f"使用设备: {device}")

    # ====== 2. 加载数据 ======
    logger.info("加载数据集...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders()

    # ====== 3. 构建模型 ======
    model = build_model()
    model = model.to(device)

    # ====== 4. 配置训练组件 ======

    # 损失函数：加权交叉熵
    # 为什么用加权交叉熵？
    # - 如果正负样本数量不均衡（如 349 vs 397），普通交叉熵会偏向多数类
    # - 加权后，少数类的每个样本贡献更大，平衡各类的影响
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"使用加权交叉熵损失，权重: {class_weights.cpu().tolist()}")

    # 优化器：AdamW
    # 只优化 requires_grad=True 的参数（冻结的层自动排除）
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    logger.info(f"使用 AdamW 优化器，lr={config.LR}, weight_decay={config.WEIGHT_DECAY}")

    # 学习率调度器：余弦退火
    # 学习率从 config.LR 平滑衰减到接近 0
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    logger.info(f"使用余弦退火学习率调度，T_max={config.EPOCHS}")

    # ====== 5. 训练循环 ======
    logger.info(f"\n开始训练，共 {config.EPOCHS} 个 epoch...")
    logger.info("=" * 60)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # 打印进度
        logger.info(
            f"Epoch {epoch:3d}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        # ====== 早停检查 + 保存最佳模型 ======
        if val_loss < best_val_loss - config.EARLY_STOP_MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0

            # 保存最佳模型
            if config.SAVE_BEST_ONLY:
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": {
                        "model_name": config.MODEL_NAME,
                        "img_size": config.IMG_SIZE,
                        "num_classes": config.NUM_CLASSES,
                        "img_mean": config.IMG_MEAN,
                        "img_std": config.IMG_STD,
                    },
                }, checkpoint_path)
                logger.info(f"  >>> 保存最佳模型 (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                logger.info(f"\n早停触发！连续 {config.EARLY_STOP_PATIENCE} 个 epoch 验证 loss 未改善。")
                logger.info(f"最佳验证 Loss: {best_val_loss:.4f}")
                break

    total_time = time.time() - start_time
    logger.info(f"\n训练完成！总用时: {total_time / 60:.1f} 分钟")

    # ====== 6. 保存训练历史 ======
    # 保存训练曲线
    plot_path = os.path.join(config.FIGURE_DIR, "training_history.png")
    utils.plot_training_history(history, save_path=plot_path)
    logger.info(f"训练曲线已保存: {plot_path}")

    # 保存 loss 和 acc 数值到文件（方便后续分析）
    import json
    history_path = os.path.join(config.LOG_DIR, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"训练历史已保存: {history_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    train()
