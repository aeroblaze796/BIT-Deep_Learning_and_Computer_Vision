# 作业一：基于PlantDoc数据集的植物病害识别 (HSV颜色 + GLCM纹理)
# 不再关注形状(HOG)，而是关注病斑的颜色(HSV)和表面纹理(GLCM)

import os
import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

warnings.filterwarnings("ignore")

from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 配置路径
TRAIN_DIR = r"D:\PlantDoc\TRAIN"
TEST_DIR = r"D:\PlantDoc\TEST"

# 1. 新的特征提取器: HSV颜色矩 + GLCM纹理

def get_color_moments(img):
    """
    提取颜色特征：将RGB转为HSV，计算 H, S, V 三个通道的均值和标准差
    HSV 能够更好地捕捉病斑颜色（如黄化、褐变）
    """
    # 转为 HSV 空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 分离通道
    h, s, v = cv2.split(hsv)
    
    # 计算一阶矩(均值)和二阶矩(标准差)
    color_features = []
    for channel in [h, s, v]:
        color_features.append(np.mean(channel)) # 颜色深浅
        color_features.append(np.std(channel))  # 颜色分布的离散程度
        
    return np.array(color_features)

def get_glcm_texture(img):
    """
    提取纹理特征：基于灰度共生矩阵 (GLCM)
    用于描述叶片表面的粗糙度、对比度等，对霉斑、锈病很敏感
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # GLCM 参数：距离1像素，角度包含0, 45, 90, 135度
    # levels=256 计算太慢，先压缩灰度级到 0-32 加快速度
    gray_quantized = (gray // 8).astype(np.uint8) 
    
    glcm = graycomatrix(gray_quantized, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=32, symmetric=True, normed=True)
    
    # 提取统计属性
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    texture_features = []
    
    for prop in props:
        # 计算该属性在4个方向上的均值
        val = graycoprops(glcm, prop).ravel()
        texture_features.append(np.mean(val))
        texture_features.append(np.std(val)) # 也加上方差信息
        
    return np.array(texture_features)

def extract_features_new(image_path, resize_shape=(128, 128)):
    """融合 HSV 和 GLCM"""
    img = cv2.imread(image_path)
    if img is None: return None
    
    try:
        # 统一尺寸
        img = cv2.resize(img, resize_shape)
        
        # 颜色特征 (6维)
        color_feat = get_color_moments(img)
        
        # 纹理特征 (12维)
        texture_feat = get_glcm_texture(img)
        
        # 拼接 (共 18 维特征 - 特征非常紧凑，训练会很快)
        return np.hstack([color_feat, texture_feat])
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# 2. 数据加载

def load_dataset_features(directory, dataset_name="Train"):
    X, y = [], []
    classes = sorted(os.listdir(directory))
    print(f"\n[{dataset_name}] 开始提取特征 (HSV+GLCM)")
    
    start_time = time.time()
    for label_name in classes:
        class_dir = os.path.join(directory, label_name)
        if not os.path.isdir(class_dir): continue
        
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png'))]
        
        for img_file in img_files:
            feat = extract_features_new(os.path.join(class_dir, img_file))
            if feat is not None:
                X.append(feat)
                y.append(label_name)
                
    print(f"[{dataset_name}] 完成。耗时: {time.time()-start_time:.2f}s, 样本数: {len(X)}")
    return np.array(X), np.array(y)

# 执行加载
print("步骤 1: 特征提取 (颜色矩 + GLCM纹理)")
X_train, y_train_labels = load_dataset_features(TRAIN_DIR, "训练集")
X_test, y_test_labels = load_dataset_features(TEST_DIR, "测试集")

print(f"新特征维度: {X_train.shape[1]} (特征越少，模型越不容易过拟合)")

print("步骤 2: 数据标准化")
le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)
y_test = le.transform(y_test_labels)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 训练与对比

print("步骤 3: 9种模型")

models_dict = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Gaussian NB": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM (Linear)": SVC(kernel='linear', cache_size=1000),
    "SVM (RBF)": SVC(kernel='rbf', cache_size=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

results = []
trained_models = {}
predictions = {}

for name, clf in models_dict.items():
    print(f"正在训练: {name} ", end="")
    t0 = time.time()
    try:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cost_time = time.time() - t0
        
        print(f"Acc: {acc:.2%} | Time: {cost_time:.2f}s")
        
        results.append({
            "Model": name, 
            "Accuracy": acc * 100, 
            "Time": cost_time
        })
        trained_models[name] = clf
        predictions[name] = y_pred
        
    except Exception as e:
        print(f"失败: {e}")

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

# 4. 可视化
print("步骤 4: 生成新图表")

# 1. 排行榜
plt.figure(figsize=(12, 7))
ax = sns.barplot(x="Accuracy", y="Model", data=results_df, palette="magma")
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f%%', padding=3)
plt.title("基于 [HSV颜色 + GLCM纹理] 的模型排行榜", fontsize=16)
plt.xlabel("准确率 (%)")
plt.tight_layout()
plt.savefig("chart_1_hsv_glcm_rank.png", dpi=150)

# 2. 最佳模型混淆矩阵
best_model_name = results_df.iloc[0]["Model"]
print(f"本次实验冠军: {best_model_name}")

plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_test, predictions[best_model_name])
sns.heatmap(cm, annot=False, cmap='Greens', fmt='d', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'混淆矩阵 ({best_model_name} - 颜色/纹理特征)', fontsize=18)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("chart_2_confusion_hsv_glcm.png", dpi=150)

# 3. 详细报告
print(f"{best_model_name} 详细报告")
print(classification_report(y_test, predictions[best_model_name], target_names=le.classes_))

with open('best_hsv_glcm_model.pkl', 'wb') as f:
    pickle.dump(trained_models[best_model_name], f)

print("实验结束，图片已保存：")
print("1. chart_1_hsv_glcm_rank.png")
print("2. chart_2_confusion_hsv_glcm.png")