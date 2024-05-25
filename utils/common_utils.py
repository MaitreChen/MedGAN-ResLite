import os
import random
import shutil


# colors = ['\033[31m', '\033[33m', '\033[32m', '\033[34m', '\033[35m', '\033[36m']  # 不同颜色的 ANSI 转义码
def print_separator():
    color = '\033[31m'  # 红色的 ANSI 转义码
    reset_color = '\033[0m'  # 重置颜色为默认值

    print(color + '**' * 40 + reset_color)  # 打印红色的彩虹颜色间隔线


def create_folder(path: str):
    """
    This function creates a folder at the specified path if it does not already exist
    """
    if not os.path.exists(path):
        os.makedirs(path)


def split_train_val(input_dir, output_dir, val_ratio=0.1):
    """
    将输入目录中的图像按照指定比例划分为训练集和验证集，并保存到输出目录中。

    参数：
    input_dir: 输入目录，包含待划分的图像文件。
    output_dir: 输出目录，保存划分后的图像文件。
    val_ratio: 验证集比例，范围为[0, 1]，默认为0.1。

    返回：
    无。
    """

    # 创建输出目录中的train和val文件夹
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 遍历输入目录中的类别文件夹
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)

        # 如果是文件夹，则处理
        if os.path.isdir(class_dir):
            # 获取类别下的图像文件列表
            image_files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

            # 计算验证集大小
            val_size = int(len(image_files) * val_ratio)

            # 随机选择验证集中的图像
            val_images = random.sample(image_files, val_size)

            # 将验证集图像移动到val文件夹中
            for image in val_images:
                src = os.path.join(class_dir, image)
                dst = os.path.join(val_dir, class_name, image)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)

            # 剩余的图像移动到train文件夹中
            for image in image_files:
                if image not in val_images:
                    src = os.path.join(class_dir, image)
                    dst = os.path.join(train_dir, class_name, image)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(src, dst)


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(project_root)

    input_dir = os.path.join(project_root, 'data', 'chest_xray2017_processed', 'train')
    output_dir = os.path.join(project_root, 'data', 'real')

    split_train_val(input_dir, output_dir, val_ratio=0.1)
