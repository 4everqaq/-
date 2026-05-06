"""
单张 CT 图像预测程序

功能：
- 输入一张肺部 CT 图像
- 输出预测结果（阳性/阴性）和置信度
- 可视化预测结果

用法：
    python predict.py --image path/to/ct_image.png
    python predict.py --image path/to/ct_image.png --model path/to/model.pth
"""
import os
import argparse
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import utils
from model import build_model


def predict_image(image_path, model_path=None, device="cpu"):
    """
    对单张 CT 图像进行预测

    Args:
        image_path: 输入图像路径
        model_path: 模型权重路径
        device: 计算设备

    Returns:
        pred_label: 预测类别 (0=阴性, 1=阳性)
        confidence: 预测置信度
        probs: 各类别概率
    """
    # 加载模型
    model = build_model()

    if model_path is None:
        model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        print("请先运行 train.py 训练模型")
        return None, None, None

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 预处理图像（与训练时验证集相同的处理方式）
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_label = probs.max(dim=1)

    pred_label = pred_label.item()
    confidence = confidence.item()
    probs = probs.squeeze(0).cpu().numpy()

    return pred_label, confidence, probs


def visualize_prediction(image_path, pred_label, confidence, probs, save_path=None):
    """
    可视化预测结果

    显示：
    - 原始 CT 图像
    - 预测结果（阴性/阳性）
    - 各类别概率
    """
    image = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左侧：原始图像
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("CT Image", fontsize=14)
    axes[0].axis("off")

    # 右侧：预测结果
    class_names = ["阴性 (Negative)", "阳性 (Positive)"]
    colors = ["#2ecc71", "#e74c3c"]  # 绿色 = 阴性，红色 = 阳性

    result_text = class_names[pred_label]
    result_color = colors[pred_label]

    axes[1].text(0.5, 0.7, f"预测结果:", fontsize=16, ha="center",
                 transform=axes[1].transAxes, fontweight="bold")
    axes[1].text(0.5, 0.55, result_text, fontsize=22, ha="center",
                 transform=axes[1].transAxes, color=result_color, fontweight="bold")
    axes[1].text(0.5, 0.42, f"置信度: {confidence * 100:.1f}%", fontsize=16,
                 ha="center", transform=axes[1].transAxes)

    # 概率柱状图
    y_pos = [0.25, 0.15]
    axes[1].barh(y_pos, probs, color=colors, alpha=0.7, height=0.08)
    for i, (p, name) in enumerate(zip(probs, class_names)):
        axes[1].text(p + 0.01, y_pos[i], f"{name}: {p * 100:.1f}%",
                     va="center", fontsize=10)

    axes[1].set_xlim(0, 1.1)
    axes[1].axis("off")

    plt.suptitle("Pneumonia CT Image Classification", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="肺炎 CT 影像单张预测")
    parser.add_argument("--image", type=str, required=True, help="CT 图像路径")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径（可选）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备 (auto/cuda/cpu)")
    parser.add_argument("--no-viz", action="store_true", help="不生成可视化图片")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"设备: {device}")
    print(f"输入图像: {args.image}")

    # 预测
    pred_label, confidence, probs = predict_image(args.image, args.model, device)

    if pred_label is None:
        return

    class_names = ["阴性 (Negative)", "阳性 (Positive)"]
    print(f"\n{'=' * 40}")
    print(f"预测结果: {class_names[pred_label]}")
    print(f"置信度: {confidence * 100:.1f}%")
    print(f"各类别概率: 阴性={probs[0] * 100:.1f}%, 阳性={probs[1] * 100:.1f}%")
    print(f"{'=' * 40}")

    # 可视化
    if not args.no_viz:
        save_path = os.path.join(config.FIGURE_DIR, "prediction_result.png")
        visualize_prediction(args.image, pred_label, confidence, probs, save_path=save_path)
        print(f"\n可视化结果已保存: {save_path}")


if __name__ == "__main__":
    main()
