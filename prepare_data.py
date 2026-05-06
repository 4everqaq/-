"""
数据集下载与准备脚本

COVID-CT 数据集来自：https://github.com/UCSD-AI4H/COVID-CT

本脚本帮助你：
1. 下载数据集（如果尚未下载）
2. 创建训练/验证/测试集的划分
3. 生成标注 txt 文件

如果你的数据集已经有 txt 标注文件，可以跳过此脚本。
"""

from pathlib import Path
import random
import config
import argparse
import config


def prepare_dataset_from_folders(data_dir):
    """
    如果你的数据集是按文件夹组织的（如 COVID/images/ 和 normal/images/），
    此函数会将其转换为项目所需的 txt 格式。

    期望的文件夹结构：
    data/COVID-CT/
    ├── COVID/
    │   └── images/
    │       ├── image1.png
    │       ├── image2.png
    │       └── ...
    └── normal/
        └── images/
            ├── image1.png
            ├── image2.png
            └── ...
    """
    data_dir = Path(data_dir)
    # 修改点1：路径改为 COVID/images 和 normal/images
    covid_dir = data_dir / "COVID" / "images"
    normal_dir = data_dir / "normal" / "images"

    samples = []

    # 读取阳性样本（COVID）
    if covid_dir.exists():
        for img in covid_dir.glob("*.png"):
            samples.append((str(img), 1))
        for img in covid_dir.glob("*.jpg"):
            samples.append((str(img), 1))

    # 读取阴性样本（normal）
    # 修改点2：non-COVID 改为 normal
    if normal_dir.exists():
        for img in normal_dir.glob("*.png"):
            samples.append((str(img), 0))
        for img in normal_dir.glob("*.jpg"):
            samples.append((str(img), 0))

    if not samples:
        print(f"在 {data_dir} 中未找到图像文件！")
        print("请确保数据集按以下结构组织:")
        print(f"  {data_dir}/COVID/images/*.png")
        print(f"  {data_dir}/normal/images/*.png")
        return False

    print(f"共找到 {len(samples)} 张图像:")
    pos_count = sum(1 for _, l in samples if l == 1)
    neg_count = len(samples) - pos_count
    print(f"  阳性 (COVID): {pos_count} 张")
    print(f"  阴性 (normal): {neg_count} 张")

    # 随机打乱并划分
    random.seed(config.SEED)
    random.shuffle(samples)

    # 80% 训练，10% 验证，10% 测试
    n = len(samples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    # 写入 txt 文件
    def write_txt(path, samples_list):
        with open(path, "w", encoding="utf-8") as f:
            for img_path, label in samples_list:
                f.write(f"{img_path},{label}\n")

    write_txt(config.TRAIN_TXT, train_samples)
    write_txt(config.VAL_TXT, val_samples)
    write_txt(config.TEST_TXT, test_samples)

    print(f"\n数据划分完成:")
    print(f"  训练集: {len(train_samples)} 张 -> {config.TRAIN_TXT}")
    print(f"  验证集: {len(val_samples)} 张 -> {config.VAL_TXT}")
    print(f"  测试集: {len(test_samples)} 张 -> {config.TEST_TXT}")

    return True


def download_covid_ct():
    """
    提供 COVID-CT 数据集的下载指引

    由于数据集较大且可能涉及版权问题，此处提供手动下载指引。
    """
    print("=" * 60)
    print("COVID-CT 数据集下载指引")
    print("=" * 60)
    print()
    print("方式一：从 GitHub 下载")
    print("  地址: https://github.com/UCSD-AI4H")
    print("  说明: 该仓库包含 COVID-CT 数据集的图像和标注信息")
    print()
    print("方式二：从 Kaggle 下载")
    print("  地址: https://www.kaggle.com/datasets/tawsifurrahman")
    print("  说明: 包含整理好的 COVID 和 normal CT 图像")
    print()
    print("下载后请将数据放在:")
    print(f"  {config.DATA_DIR}/")
    print()
    print("如果你的数据是按文件夹组织的 (COVID/images/ 和 normal/images/)，")
    print("运行以下命令自动划分:")
    print(f"  python prepare_data.py --folder {config.DATA_DIR}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备 COVID-CT 数据集")
    parser.add_argument("--folder", type=str, default=None,
                        help="数据集文件夹路径（如果按文件夹组织）")
    parser.add_argument("--download-info", action="store_true",
                        help="显示下载信息")
    args = parser.parse_args()

    if args.download_info:
        download_covid_ct()
    elif args.folder:
        prepare_dataset_from_folders(args.folder)
    else:
        download_covid_ct()
        print()
        prepare_dataset_from_folders(config.DATA_DIR)