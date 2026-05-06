"""
肺炎CT影像识别系统 - 超参数配置文件
所有可调参数集中在此，方便统一管理和调优
"""
import os

# ====== 路径配置 ======
# 数据集根目录，根据实际情况修改
DATA_DIR = r"data/COVID-CT"

# 数据集中训练集、验证集、测试集的 txt 文件路径
# COVID-CT 数据集通常提供这些文件来标注每张图片的类别
TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
VAL_TXT = os.path.join(DATA_DIR, "val.txt")
TEST_TXT = os.path.join(DATA_DIR, "test.txt")

# 如果数据集没有划分 val/test，可以只用 train，代码会自动从 train 中切分
USE_AUTO_SPLIT = True  # 设为 True 时自动从训练集划分 20% 作为验证集

# 结果输出目录
RESULTS_DIR = "results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
FIGURE_DIR = os.path.join(RESULTS_DIR, "figures")

# ====== 数据配置 ======
IMG_SIZE = 224           # 输入图像尺寸，ResNet 标准输入
BATCH_SIZE = 16          # 批次大小，根据显存调整（8-32 均可）
NUM_WORKERS = 4          # 数据加载线程数
NUM_CLASSES = 2          # 二分类：肺炎阳性 / 阴性

# ImageNet 归一化参数（与预训练权重保持一致）
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# ====== 模型配置 ======
MODEL_NAME = "resnet50"          # 可选: resnet18, resnet34, resnet50, resnet101
PRETRAINED = True                # 是否使用 ImageNet 预训练权重
DROPOUT_RATE = 0.5               # 全连接层前的 Dropout 比率
FREEZE_STAGES = 2                # 冻结前 N 个 stage 的参数（0=不冻结）

# ====== 训练配置 ======
EPOCHS = 60              # 最大训练轮数（有早停保护，可以设大）
LR = 1e-4                # 初始学习率
WEIGHT_DECAY = 1e-4      # AdamW 的权重衰减系数（L2 正则化）

# ====== 早停配置 ======
EARLY_STOP_PATIENCE = 15         # 验证集 loss 连续 N 轮不下降则停止
EARLY_STOP_MIN_DELTA = 1e-4      # loss 下降的最小变化量（小于此值视为无改善）

# ====== 其他配置 ======
SEED = 42                # 随机种子，保证实验可复现
DEVICE = "auto"          # "auto" 自动选择 cuda/cpu，也可指定 "cuda" 或 "cpu"
SAVE_BEST_ONLY = True    # 是否只保存验证集上表现最好的模型
