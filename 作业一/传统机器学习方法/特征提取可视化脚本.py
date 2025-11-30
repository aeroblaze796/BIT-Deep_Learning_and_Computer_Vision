import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, graycomatrix, graycoprops

# 1. 配置路径
TRAIN_DIR = r".\TRAIN"
# 设置matplotlib以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 随机选择一张图片
def get_random_image(directory):
    """从数据集中随机选择一个图片文件路径"""
    if not os.path.exists(directory):
        print(f"目录 '{directory}' 不存在。")
        return None
        
    all_classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if not all_classes:
        print(f"目录 '{directory}' 中没有找到子目录。")
        return None

    random_class = random.choice(all_classes)
    class_path = os.path.join(directory, random_class)
    
    all_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not all_images:
        print(f"目录 '{class_path}' 中没有找到图片。")
        return None
        
    random_image_name = random.choice(all_images)
    image_path = os.path.join(class_path, random_image_name)
    
    print(f"已随机选择图片{image_path}")
    return image_path, random_class

# 3. 可视化函数

# HOG 可视化
def visualize_hog(img):
    """计算并返回 HOG 特征的可视化图像"""
    img_resized = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, hog_image = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return hog_image

# HSV 可视化
def visualize_hsv(img):
    """返回 HSV 空间的 H, S, V 三个通道的图像"""
    img_resized = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return h, s, v

# GLCM 可视化
def visualize_glcm(img):
    """返回 GLCM 计算前的量化图像和 GLCM 矩阵本身的热力图"""
    img_resized = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 1. 量化图像: 这是GLCM实际“看到”的输入
    # 将256个灰度级压缩到8个，以突出纹理结构
    quantized_gray = (gray // 32).astype(np.uint8)
    
    # 2. 计算GLCM矩阵
    # 只计算一个方向（0度）用于可视化
    glcm = graycomatrix(quantized_gray, distances=[1], angles=[0], levels=8,
                        symmetric=True, normed=True)
    
    return quantized_gray, glcm[:, :, 0, 0]

# SIFT 可视化
def visualize_sift(img):
    """在原图上绘制 SIFT 关键点"""
    img_resized = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)
    
    # 在彩色图上绘制关键点
    sift_image = cv2.drawKeypoints(img_resized, keypoints, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return sift_image

# 4. 主流程
if __name__ == "__main__":
    image_path, class_name = get_random_image(TRAIN_DIR)
    
    if image_path:
        original_img = cv2.imread(image_path)
        
        # 执行所有可视化
        hog_viz = visualize_hog(original_img.copy())
        h_viz, s_viz, v_viz = visualize_hsv(original_img.copy())
        glcm_input_viz, glcm_matrix_viz = visualize_glcm(original_img.copy())
        sift_viz = visualize_sift(original_img.copy())
        
        # 准备绘图
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        plt.suptitle(f"特征提取方法可视化\n类别: {class_name}", fontsize=24)

        # 转换 BGR 到 RGB 用于 matplotlib 显示
        original_img_rgb = cv2.cvtColor(cv2.resize(original_img, (256, 256)), cv2.COLOR_BGR2RGB)
        sift_viz_rgb = cv2.cvtColor(sift_viz, cv2.COLOR_BGR2RGB)

        # 填充子图
        # 1. 原图
        axes[0, 0].imshow(original_img_rgb)
        axes[0, 0].set_title("1. 原始图片", fontsize=16)
        
        # 2. HOG
        axes[0, 1].imshow(hog_viz, cmap='gray')
        axes[0, 1].set_title("2. HOG 可视化 (梯度方向)", fontsize=16)

        # 3. SIFT
        axes[0, 2].imshow(sift_viz_rgb)
        axes[0, 2].set_title("3. SIFT 可视化 (关键点)", fontsize=16)

        # 4. HSV - H
        axes[1, 0].imshow(h_viz, cmap='hsv')
        axes[1, 0].set_title("4a. HSV - Hue (色相)", fontsize=16)
        
        # 5. HSV - S
        axes[1, 1].imshow(s_viz, cmap='gray')
        axes[1, 1].set_title("4b. HSV - Saturation (饱和度)", fontsize=16)
        
        # 6. HSV - V
        axes[1, 2].imshow(v_viz, cmap='gray')
        axes[1, 2].set_title("4c. HSV - Value (明度)", fontsize=16)
        
        # 7. GLCM 输入
        axes[2, 0].imshow(glcm_input_viz, cmap='gray')
        axes[2, 0].set_title("5a. GLCM 输入 (量化灰度图)", fontsize=16)
        
        # 8. GLCM 矩阵
        im = axes[2, 1].imshow(glcm_matrix_viz, cmap='viridis')
        axes[2, 1].set_title("5b. GLCM 矩阵热力图 (0°)", fontsize=16)
        fig.colorbar(im, ax=axes[2, 1])

        # 9. 隐藏最后一个空的子图
        axes[2, 2].axis('off')

        # 美化与显示
        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("特征提取可视化.png", dpi=150)
        print("可视化图片已保存为: 特征提取可视化.png")
        plt.show()