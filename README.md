# 肺炎 CT 影像识别系统

> **《深度学习程序设计》期末大作业 — 第 26 题**

基于 ResNet 卷积神经网络的肺部 CT 影像肺炎自动检测系统。使用迁移学习技术，通过 PyTorch 框架训练深度学习模型，对 CT 图像进行二分类（阳性/阴性）判断，并提供 Grad-CAM 可视化诊断依据。

---

## 🎯 项目特点

- 🏥 **医学影像分析**：针对肺部 CT 影像的肺炎自动检测
- 🤖 **深度学习模型**：ResNet50 预训练 + 迁移学习
- 🔍 **可解释性 AI**：Grad-CAM 热力图可视化病灶区域
- 📊 **Web 可视化界面**：Gradio 驱动的交互式诊断平台
- 📈 **完整评估体系**：混淆矩阵、ROC 曲线、性能指标量化
- 🎨 **图像处理增强**：支持亮度、对比度、锐度等调节

---

## 🚀 快速开始

```bash
# 1. 克隆项目
git clone <repository-url>
cd pneumonia_ct_classifier

# 2. 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt

# 4. 准备数据集（见下方说明）

# 5. 训练模型
python train.py

# 6. Web 可视化界面
python app.py
# 然后在浏览器打开 http://localhost:7860
```

---

## 📦 环境依赖

### requirements.txt

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
gradio>=4.0.0
matplotlib>=3.7.0
Pillow>=9.5.0
seaborn>=0.12.0
scikit-learn>=1.2.0
pandas>=2.0.0
opencv-python>=4.8.0
pycocotools>=2.0.6
```

### 系统要求

- Python 3.8+
- CUDA 11.8+（建议使用 GPU 训练）
- 至少 4GB 显存（BATCH_SIZE=8）
- 约 10GB 可用空间用于数据集

---

## 📁 数据集准备

### COVDID-CT 数据集

本项目使用 [COVID-CT](https://github.com/UCSD-AI4H/COVID-CT) 公开数据集。

#### 下载方式一：Git Clone

```bash
git clone https://github.com/UCSD-AI4H/COVID-CT.git
mkdir -p data/COVID-CT
# 将 COVID-CT/COVID-CT 文件夹下的内容复制到 data/COVID-CT/
```

#### 下载方式二：手动下载

从 [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-ct-dataset) 或其他镜像站下载。

### 数据标注格式

使用 `prepare_data.py` 生成标注文件 `data/train_labels.txt`：

```python
from prepare_data import prepare_dataset

prepare_dataset(
    data_dir="data/COVID-CT",
    output_file="data/train_labels.txt"
)
```

标注文件格式：
```
data/COVID-CT/positive/xxx_0.png,1
data/COVID-CT/negative/yyy_0.png,0
...
```

其中：
- `0`: 阴性（无肺炎）
- `1`: 阳性（有肺炎/COVID-19）

---

## 🔧 使用方法

### 1. 配置参数

编辑 [`config.py`](config.py)，根据实际环境修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DATA_DIR` | `"data/train_labels.txt"` | 标注文件路径 |
| `IMAGES_DIR` | `"data/COVID-CT"` | 图片目录根路径 |
| `BATCH_SIZE` | `8` | 批次大小（显存不足则调小） |
| `EPOCHS` | `30` | 训练轮数 |
| `LEARNING_RATE` | `1e-4` | 学习率 |
| `NUM_WORKERS` | `4` | 数据加载线程数 |
| `DEVICE` | `"auto"` | 设备选择 (auto/cpu/cuda) |

### 2. 训练模型

```bash
python train.py
```

**训练输出**：
- ✅ 最佳模型：`results/checkpoints/best_model.pth`
- ✅ 训练曲线：`results/figures/training_history.png`
- ✅ 训练日志：`results/logs/training.log`

**训练监控特性**：
- ⏸️ 早停机制（patience=15）自动找到最佳训练点
- 💾 每轮自动保存最佳 checkpoint
- 📊 实时显示损失曲线、准确率
- 🔄 余弦退火学习率调度平滑衰减

### 3. 模型评估

```bash
python evaluate.py
```

**评估输出**：
- 🔲 混淆矩阵：`results/figures/confusion_matrix.png`
- 📉 ROC 曲线：`results/figures/roc_curve.png`
- 📋 性能汇总：`results/figures/metrics_summary.txt`

### 4. 单张图像预测

```bash
python predict.py --image path/to/image.png
python predict.py --image img.png --output result.png
```

### 5. Web 可视化界面（推荐）

```bash
python app.py
```

访问 **http://localhost:7860**

**Web 界面功能概览**：

| 功能 | 描述 |
|------|------|
| 🔍 单张预测 | 上传 CT 图像实时诊断 |
| 📦 批量预测 | 同时处理多张图像 |
| 🎨 Grad-CAM 热力图 | 可视化模型关注的病灶区域 |
| ⚙️ 图像增强 | 亮度、对比度、锐度、平滑度调节 |
| 🎯 预设方案 | 6 种快速图像优化预设 |
| 📊 对比视图 | 原图与热力图并排对比 |
| 📄 导出报告 | 支持 PDF 和图片格式 |
| 📈 模型性能 | 训练曲线、ROC 曲线、混淆矩阵 |
| 📋 历史记录 | 保存和管理预测记录 |
| 💡 诊断建议 | 根据预测结果给出建议 |

### 6. Web 可视化界面（推荐）

```bash
python app.py
```

访问 **http://localhost:7860**

Web 界面集成了以下功能：
- 🔍 单张预测 & 📦 批量预测
- 🎨 Grad-CAM 热力图可视化
- ⚙️ 图像增强（亮度/对比度/锐度/平滑度）
- 📊 模型性能指标展示
- 📋 预测历史记录管理
- 📄 PDF/图片报告导出

如需通过代码集成，可参考 [`app.py`](app.py) 中的预测函数：

```python
import torch
from model import build_model
from gradcam import GradCAM, visualize_cam
from PIL import Image
import torchvision.transforms as transforms
import config

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model()
model_path = "results/checkpoints/best_model.pth"
checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device).eval()

# 预处理函数
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor, image

# 预测函数
def predict(image_path, model, device, alpha=0.4):
    input_tensor, original_image = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_label = probs.max(dim=1)
    
    # 生成 Grad-CAM
    gradcam = GradCAM(model, target_layer="backbone.layer4")
    cam = gradcam.generate_cam(input_tensor, pred_label.item())
    
    result_image = visualize_cam(original_image, cam, 
                                  pred_label.item(), confidence.item())
    
    return {
        'label': '阳性' if pred_label.item() == 1 else '阴性',
        'confidence': confidence.item(),
        'probs': probs.squeeze().cpu().numpy().tolist(),
        'image': result_image
    }

# 使用示例
result = predict("test.jpg", model, device)
print(f"预测结果：{result['label']}, 置信度：{result['confidence']*100:.2f}%")
result['image'].save("output.png")
```

---

## 📂 项目结构

```
pneumonia_ct_classifier/
├── config.py          # 超参数配置
├── dataset.py         # 数据集加载与预处理
├── model.py           # ResNet 模型构建
├── train.py           # 训练主程序
├── evaluate.py        # 模型评估
├── predict.py         # 单张图像预测命令行
├── utils.py           # 工具函数
├── gradcam.py         # Grad-CAM 可视化模块
├── prepare_data.py    # 数据预处理脚本
├── app.py             # Web 可视化界面（Gradio）
│   ├── load_model()              # 加载模型
│   ├── predict()                 # 单张预测核心函数
│   ├── batch_predict()           # 批量预测
│   └── export_current_report()   # 导出报告
├── requirements.txt   # 依赖清单
├── README.md          # 本文件
├── 设计文档.md        # 系统设计详细说明
├── 肺炎 CT 影像分类系统_大作业论文.md  # 完整的技术论文
└── results/           # 训练结果输出目录
    ├── checkpoints/   # 模型权重 (.pth)
    ├── logs/          # 训练日志 (.log/.json)
    ├── figures/       # 可视化图表 (.png/.jpg)
    └── reports/       # 诊断报告 (.pdf)
```

---

## 🛠️ 技术架构

### 模型架构

| 组件 | 选择 | 说明 |
|------|------|------|
| **基础模型** | ResNet50 (预训练) | ImageNet 权重迁移学习 |
| **输入尺寸** | 224×224 | 标准网络输入尺寸 |
| **通道数** | 3 | RGB 三通道 |
| **输出层** | Fully Connected → 2 | 二分类输出 |

### 训练策略

| 组件 | 配置 | 作用 |
|------|------|------|
| **损失函数** | 加权交叉熵 | 处理阴阳性样本不平衡 |
| **优化器** | AdamW | lr=1e-4, weight_decay=1e-4 |
| **学习率调度** | 余弦退火 CosineAnnealingLR | 平滑衰减避免局部最优 |
| **数据增强** | RandomResizedCrop + RandomHorizontalFlip + Rotate | 防止过拟合提升泛化能力 |
| **早停机制** | patience=15 epochs | 自动找到最佳训练点 |

### 评估指标

| 指标 | 公式/说明 |
|------|----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) 整体准确率 |
| **Precision** | TP/(TP+FP) 查准率 |
| **Recall** | TP/(TP+FN) 召回率 |
| **F1-Score** | 2×Precision×Recall/(Precision+Recall) |
| **AUC-ROC** | ROC 曲线下面积 |

---

## 📊 预期性能指标

| 指标 | 目标范围 |
|------|---------|
| Accuracy | 80%-90% |
| Recall | 75%-90% |
| AUC | 0.85-0.95 |
| F1-Score | 0.75-0.85 |

---

## ⚠️ 重要声明

> **⚠️ 仅供学术研究和教学演示使用，不能替代专业医疗诊断！**
>
> 本系统为深度学习教学项目，模型性能和可靠性未经临床验证。
> 任何医疗诊断决策应咨询专业医疗机构和执业医师。
>
> 作者不对本系统产生的任何医疗后果承担责任。

---

## 📚 相关资源

- **设计文档**: [`设计文档.md`](设计文档.md) - 详细的系统设计说明
- **学术论文**: [`肺炎 CT 影像分类系统_大作业论文.md`](肺炎 CT 影像分类系统_大作业论文.md) - 完整的技术论文
- **原始数据集**: [UCSD COVID-CT GitHub](https://github.com/UCSD-AI4H/COVID-CT)
- **Grad-CAM 论文**: [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1610.02391)

---

## 🤝 致谢

感谢以下开源项目的支持和启发：

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [torchvision](https://pytorch.org/vision/stable/index.html) - 视觉工具包
- [Gradio](https://www.gradio.app/) - Web UI 框架
- [COVID-CT Dataset](https://github.com/UCSD-AI4H/COVID-CT) - 公开数据集

---

<div align="center">
<sup>深度学习程序设计 2024 秋季学期 · 第 26 小组</sup>
</div>
