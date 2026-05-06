"""
模型评估程序

在测试集上全面评估训练好的模型，输出：
- Accuracy, Precision, Recall, F1-Score, Specificity
- AUC-ROC 曲线
- 混淆矩阵
- 分类报告

为什么需要独立的评估程序？
1. 训练时只监控 Loss 和 Accuracy，不够全面
2. 医学诊断需要特别关注 Recall（不能漏诊）
3. 混淆矩阵能直观展示模型在各类别上的表现
4. ROC-AUC 不受分类阈值影响，评估模型整体能力
"""
import os
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import torch.nn.functional as F

import config
import utils
from model import build_model
from dataset import get_dataloaders


@torch.no_grad()
def evaluate_on_loader(model, loader, device):
    """
    在指定 DataLoader 上进行全面评估

    Returns:
        y_true: 所有样本的真实标签
        y_pred: 所有样本的预测标签
        y_scores: 所有样本预测为正类的概率（用于 ROC 曲线）
    """
    model.eval()

    y_true = []
    y_pred = []
    y_scores = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        probs = F.softmax(outputs, dim=1)  # 转为概率分布

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(probs.argmax(dim=1).cpu().numpy())
        y_scores.extend(probs[:, 1].cpu().numpy())  # 正类概率

    return np.array(y_true), np.array(y_pred), np.array(y_scores)


def evaluate(checkpoint_path=None):
    """主评估流程"""
    # 1. 设置
    utils.set_seed(config.SEED)

    if config.DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.DEVICE)

    os.makedirs(config.FIGURE_DIR, exist_ok=True)

    # 2. 加载数据
    print("加载测试数据...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders()

    # 优先使用测试集，没有则用验证集
    eval_loader = test_loader if test_loader else val_loader
    dataset_name = "Test Set" if test_loader else "Validation Set"
    print(f"Use {dataset_name} for evaluation ({len(eval_loader.dataset)} images)")

    # 3. 加载模型
    model = build_model()

    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"错误：找不到模型文件 {checkpoint_path}")
        print("请先运行 train.py 训练模型")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"\n加载模型: {checkpoint_path}")
    print(f"训练时最佳验证 Accuracy: {checkpoint['val_acc']:.4f}")

    # 4. 执行评估
    print(f"\n正在评估模型...")
    y_true, y_pred, y_scores = evaluate_on_loader(model, eval_loader, device)

    # 5. 输出结果
    class_names = ["Negative", "Positive"]

    # 打印分类报告
    utils.print_classification_report_metrics(y_true, y_pred, class_names)

    # 保存所有指标
    metrics_path = os.path.join(config.FIGURE_DIR, "metrics_summary.txt")
    metrics = utils.save_metrics_summary(y_true, y_pred, y_scores, save_path=metrics_path)

    print(f"\n关键指标汇总:")
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")

    # 6. 可视化
    # 混淆矩阵
    cm_path = os.path.join(config.FIGURE_DIR, "confusion_matrix.png")
    utils.plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)
    print(f"\n混淆矩阵已保存: {cm_path}")

    # ROC 曲线
    roc_path = os.path.join(config.FIGURE_DIR, "roc_curve.png")
    roc_auc = utils.plot_roc_curve(y_true, y_scores, save_path=roc_path)
    print(f"ROC 曲线已保存: {roc_path}")

    print(f"\n评估完成！所有图表保存在: {config.FIGURE_DIR}")


if __name__ == "__main__":
    evaluate()
