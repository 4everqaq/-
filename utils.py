"""
工具函数模块
包含：日志记录、可视化、随机种子设置等辅助功能
"""
import os
import random
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，服务器也能用
import matplotlib.pyplot as plt

# 设置中文字体（优先使用系统字体）
import matplotlib.font_manager as fm
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 尝试设置中文字体，如果没有则忽略警告
try:
    font_path = fm.findfont('SimHei') or fm.findfont('Microsoft YaHei')
except:
    pass
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
import config


def set_seed(seed=42):
    """
    设置全局随机种子，保证实验结果可复现

    为什么要设置随机种子？
    - 深度学习中有大量随机操作：权重初始化、dropout、数据打乱等
    - 固定种子后，每次运行得到相同结果，方便调试和对比实验
    - 需要同时设置 Python、NumPy、PyTorch 的随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证 CuDNN 的确定性（可能略微降低速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name="pneumonia_ct", log_file=None, level=logging.INFO):
    """
    配置日志系统，同时输出到控制台和文件

    Args:
        name: logger 名称
        log_file: 日志文件路径
        level: 日志级别
    Returns:
        logging.Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # 清除已有 handlers

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 文件 handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def plot_training_history(history, save_path=None):
    """
    绘制训练过程中的 Loss 和 Accuracy 曲线

    Args:
        history: dict，包含 train_loss, val_loss, train_acc, val_acc
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 曲线
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    绘制混淆矩阵热力图

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        save_path: 保存路径
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names or ["Negative", "Positive"],
                yticklabels=class_names or ["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_scores, save_path=None):
    """
    绘制 ROC 曲线并计算 AUC

    Args:
        y_true: 真实标签 (0/1)
        y_scores: 模型预测为正类的概率
        save_path: 保存路径
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return roc_auc


def print_classification_report_metrics(y_true, y_pred, class_names=None):
    """
    打印详细的分类报告（Precision, Recall, F1）
    """
    report = classification_report(
        y_true, y_pred,
        target_names=class_names or ["Negative", "Positive"],
        zero_division=0
    )
    print("\n" + "=" * 50)
    print("Classification Report:")
    print("=" * 50)
    print(report)
    return report


def save_metrics_summary(y_true, y_pred, y_scores, save_path=None):
    """
    保存所有评估指标到文本文件
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }
    # 计算特异性 (Specificity) = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["Specificity"] = specificity

    roc_auc = auc(roc_curve(y_true, y_scores)[0], roc_curve(y_true, y_scores)[1])
    metrics["AUC-ROC"] = roc_auc

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("Pneumonia CT Classifier - Evaluation Metrics Summary\n")
            f.write("=" * 50 + "\n\n")
            for name, val in metrics.items():
                f.write(f"{name}: {val:.4f}\n")
            f.write("\n")
            f.write(classification_report(
                y_true, y_pred,
                target_names=["Negative", "Positive"],
                zero_division=0
            ))

    return metrics
