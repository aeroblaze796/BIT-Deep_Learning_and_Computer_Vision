import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

# 1. 配置
# 随机从 'TRAIN/Apple Scab Leaf' 文件夹中选一张
try:
    image_folder = 'TRAIN/Apple Scab Leaf'
    image_name = random.choice(os.listdir(image_folder))
    IMAGE_PATH = os.path.join(image_folder, image_name)
except FileNotFoundError:
    print(f"找不到文件夹 '{image_folder}'。")
    IMAGE_PATH = None 

# 每种增强策略生成多少个样本进行展示
NUM_SAMPLES_TO_SHOW = 5

# 2. 定义与实验中完全相同的数据增强策略
def get_augmentation_pipelines():
    """定义三种强度的数据增强方案。"""
    pipelines = {
        '弱增强 (Gentle)': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # 为了可视化，暂时不进行ToTensor和Normalize
        ]),
        '中等增强 (Moderate)': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]),
        '强增强 (Aggressive)': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            # RandomErasing需要Tensor输入，在这里单独处理
            # 为了简单起见，在可视化中主要展示几何和颜色变换
        ])
    }
    # 单独定义RandomErasing，因为它需要在Tensor上操作
    random_erasing_transform = transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
    
    return pipelines, random_erasing_transform

# 3. 主可视化函数
def visualize_augmentations(image_path):
    if not image_path or not os.path.exists(image_path):
        print(f"图片路径 '{image_path}' 不存在。")
        return

    # 加载原图
    original_img = Image.open(image_path).convert('RGB')
    
    pipelines, random_erasing_transform = get_augmentation_pipelines()
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建一个大的画布来展示所有图片
    # 4行: 原图, 弱, 中, 强
    # NUM_SAMPLES_TO_SHOW + 1列: 标签 + 样本
    fig, axs = plt.subplots(4, NUM_SAMPLES_TO_SHOW + 1, figsize=(15, 10))

    # 第一行: 展示原图
    axs[0, 0].text(0.5, 0.5, '原图\n(Original)', ha='center', va='center', fontsize=15)
    axs[0, 0].axis('off')
    axs[0, 1].imshow(original_img)
    axs[0, 1].set_title('原始图片')
    axs[0, 1].axis('off')
    for j in range(2, NUM_SAMPLES_TO_SHOW + 1):
        axs[0, j].axis('off')

    # 循环处理每一种增强策略
    for i, (name, pipeline) in enumerate(pipelines.items()):
        row_idx = i + 1
        
        # 显示策略名称
        axs[row_idx, 0].text(0.5, 0.5, name, ha='center', va='center', fontsize=15)
        axs[row_idx, 0].axis('off')

        # 生成并显示增强后的样本
        for j in range(NUM_SAMPLES_TO_SHOW):
            augmented_img = pipeline(original_img)
            
            # 特别处理强增强中的RandomErasing
            if name.startswith('强增强'):
                # 转换到Tensor，应用擦除，再转回PIL Image以供显示
                img_tensor = transforms.ToTensor()(augmented_img)
                erased_tensor = random_erasing_transform(img_tensor)
                augmented_img = transforms.ToPILImage()(erased_tensor)

            axs[row_idx, j+1].imshow(augmented_img)
            axs[row_idx, j+1].axis('off')

    plt.tight_layout()
    plt.suptitle(f'不同数据增强策略效果对比\n(原图: {os.path.basename(image_path)})', fontsize=20)
    fig.subplots_adjust(top=0.9)
    plt.show()


# 4. 运行脚本
if __name__ == '__main__':
    visualize_augmentations(IMAGE_PATH)