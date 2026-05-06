"""
Grad-CAM (Gradient-weighted Class Activation Mapping) 可视化模块

功能：
1. 生成类激活热力图，展示模型预测时关注的图像区域
2. 对于肺炎CT图像，热力图通常高亮显示病灶区域
3. 增强模型的可解释性，帮助医生理解模型的决策依据

原理：
- 利用目标类别的梯度信息
- 计算卷积层特征图的重要性权重
- 生成加权特征激活热力图
- 红色区域 = 模型认为重要的区域（潜在病灶）

参考文献：
Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
ICCV 2017
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


class GradCAM:
    """
    Grad-CAM类激活可视化

    用于生成模型预测时关注的区域热力图
    """

    def __init__(self, model, target_layer="backbone.layer4"):
        """
        Args:
            model: 神经网络模型
            target_layer: 目标卷积层名称（通常是最后一个卷积层）
        """
        self.model = model
        self.target_layer = target_layer

        # 存储特征图和梯度
        self.features = None
        self.gradients = None

        # 注册钩子函数
        self._register_hooks()

    def _register_hooks(self):
        """注册forward和backward钩子，捕获特征图和梯度"""

        def forward_hook(module, input, output):
            self.features = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # 找到目标层
        target_module = self._find_target_layer()

        if target_module is None:
            raise ValueError(f"找不到目标层: {self.target_layer}")

        # 注册钩子
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)

    def _find_target_layer(self):
        """根据名称字符串找到目标层模块"""
        parts = self.target_layer.split(".")
        module = self.model

        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None

        return module

    def generate_cam(self, input_tensor, target_class=None):
        """
        生成类激活热力图

        Args:
            input_tensor: 输入图像tensor (1, C, H, W)
            target_class: 目标类别索引（默认使用预测类别）

        Returns:
            cam: 热力图numpy数组 (H, W)，值域[0, 1]
        """
        self.model.eval()

        # 如果未指定目标类别，使用预测类别
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

        # 前向传播
        output = self.model(input_tensor)

        # 构建目标类别的one-hot向量
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        # 反向传播，计算梯度
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        # 计算Grad-CAM
        # 1. 对梯度在空间维度上全局平均池化，得到各特征图的权重
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # 2. 加权求和特征图
        cam = (weights * self.features).sum(dim=1, keepdim=True)  # (1, 1, H', W')

        # 3. ReLU激活，只保留正面影响
        cam = F.relu(cam)

        # 4. 归一化到[0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # 5. 上采样到输入图像尺寸
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))

        return cam


def visualize_cam(original_image, cam, pred_label, confidence):
    """
    可视化Grad-CAM热力图

    创建包含原图、热力图、叠加图和预测结果的组合图像

    Args:
        original_image: 原始PIL图像
        cam: Grad-CAM热力图numpy数组
        pred_label: 预测类别 (0=阴性, 1=阳性)
        confidence: 预测置信度

    Returns:
        result_image: 组合可视化结果PIL图像
    """
    # 转为numpy
    original_np = np.array(original_image.resize((224, 224)))

    # 热力图颜色映射
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 融合原图和热力图
    alpha = 0.4
    overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap, alpha, 0)

    # 创建组合图
    fig_width = 800
    fig_height = 300
    result = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)

    # 填充原图
    result[:, :224, :] = original_np

    # 填充热力图
    result[:, 224:448, :] = heatmap

    # 填充叠加图
    result[:, 448:672, :] = overlay

    # 添加文字说明（右侧区域）
    # 使用OpenCV绘制文字
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 预测结果
    class_names = ["Negative", "Positive"]
    colors = [(0, 200, 0), (200, 0, 0)]  # 绿色/红色

    # 绘制标题
    cv2.putText(result, "Grad-CAM Visualization", (690, 30), font, 0.6, (255, 255, 255), 2)

    # 绘制预测类别
    pred_color = colors[pred_label]
    cv2.putText(result, f"Result: {class_names[pred_label]}",
                (690, 80), font, 0.7, pred_color, 2)

    # 绘制置信度
    cv2.putText(result, f"Confidence: {confidence*100:.1f}%",
                (690, 120), font, 0.6, (255, 255, 255), 2)

    # 绘制区域标签
    cv2.putText(result, "Original", (60, 280), font, 0.5, (255, 255, 255), 1)
    cv2.putText(result, "Heatmap", (284, 280), font, 0.5, (255, 255, 255), 1)
    cv2.putText(result, "Overlay", (508, 280), font, 0.5, (255, 255, 255), 1)

    # 绘制分隔线
    cv2.line(result, (224, 0), (224, 280), (128, 128, 128), 2)
    cv2.line(result, (448, 0), (448, 280), (128, 128, 128), 2)
    cv2.line(result, (672, 0), (672, 280), (128, 128, 128), 2)

    # 转为PIL图像
    result_image = Image.fromarray(result)

    return result_image


def generate_gradcam_batch(model, images, target_layer="backbone.layer4"):
    """
    批量生成Grad-CAM热力图

    Args:
        model: 模型
        images: 图像tensor (N, C, H, W)
        target_layer: 目标卷积层

    Returns:
        cams: 热力图列表
    """
    gradcam = GradCAM(model, target_layer)

    cams = []
    for i in range(images.shape[0]):
        cam = gradcam.generate_cam(images[i:i+1])
        cams.append(cam)

    return cams


def visualize_cam_enhanced(original_image, cam, pred_label, confidence, alpha=0.4):
    """
    增强的 Grad-CAM 可视化 - 支持透明度调节和更高质量的渲染
    """
    original_np = np.array(original_image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap, alpha, 0)

    fig_width, fig_height = 900, 320
    result = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)
    result[:, :, :] = 30

    result[20:244, 20:244, :] = original_np
    result[20:244, 264:488, :] = heatmap
    result[20:244, 508:732, :] = overlay

    result_pil = Image.fromarray(result)
    draw = ImageDraw.Draw(result_pil)

    try:
        try:
            title_font = ImageFont.truetype("arial.ttf", 18)
            label_font = ImageFont.truetype("arial.ttf", 14)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            title_font = label_font = small_font = ImageFont.load_default()

        draw.text((750, 30), "Grad-CAM Visualization", fill=(255, 255, 255), font=title_font)
        pred_color = (255, 80, 80) if pred_label == 1 else (80, 255, 80)
        draw.text((750, 70), f"Result: {'Positive' if pred_label == 1 else 'Negative'}", fill=pred_color, font=label_font)
        draw.text((750, 100), f"Confidence: {confidence*100:.1f}%", fill=(255, 255, 255), font=label_font)
        draw.text((50, 290), "Original", fill=(200, 200, 200), font=small_font)
        draw.text((280, 290), "Heatmap", fill=(200, 200, 200), font=small_font)
        draw.text((530, 290), "Overlay", fill=(200, 200, 200), font=small_font)
        draw.text((750, 240), f"Alpha: {alpha}", fill=(150, 150, 150), font=small_font)
    except:
        pass

    return result_pil