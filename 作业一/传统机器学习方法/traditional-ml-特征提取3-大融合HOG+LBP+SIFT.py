# 作业一：基于PlantDoc数据集的植物病害识别 (传统特征融合)
# 特征工程：SIFT + HOG + LBP + SURF (尝试) -> 融合 -> 机器学习分类

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

from skimage.feature import hog, local_binary_pattern
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

# 1. 定义各种特征提取器

def get_hog_features(img, resize_shape=(128, 128)):
    """提取 HOG (方向梯度直方图) - 形状特征"""
    img_resized = cv2.resize(img, resize_shape)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    # pixels_per_cell 越大特征越少，速度越快
    features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

def get_lbp_features(img, resize_shape=(128, 128)):
    """提取 LBP (局部二值模式) - 纹理特征"""
    img_resized = cv2.resize(img, resize_shape)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    # LBP 参数
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    # 计算 LBP 直方图作为特征向量
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    # 归一化
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def get_sift_features(img, vector_size=128):
    """提取 SIFT (尺度不变特征变换) - 关键点特征"""
    # 既然不需要空间信息，可以resize小一点加快速度
    img_resized = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    try:
        sift = cv2.SIFT_create()
        kps, des = sift.detectAndCompute(gray, None)
        
        if des is None:
            # 如果没检测到特征点，返回零向量
            return np.zeros(vector_size * 2)
        
        # 简化策略：计算所有特征描述子的均值和标准差
        # 这样无论有多少个特征点，最终向量长度都是固定的 (128*2 = 256)
        mu = np.mean(des, axis=0)
        std = np.std(des, axis=0)
        return np.hstack([mu, std])
        
    except Exception:
        return np.zeros(vector_size * 2)

def get_surf_features(img, vector_size=64):
    """提取 SURF (加速稳健特征) - 尝试提取"""
    # SURF 是专利算法，在新版 OpenCV (4.x+) 中通常被移除了
    # 这里加个保护，如果环境不支持 SURF，自动返回空特征，不报错
    try:
        if hasattr(cv2, 'xfeatures2d'):
            surf = cv2.xfeatures2d.SURF_create()
            img_resized = cv2.resize(img, (256, 256))
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            kps, des = surf.detectAndCompute(gray, None)
            
            if des is None:
                return np.zeros(vector_size * 2)
            
            mu = np.mean(des, axis=0)
            std = np.std(des, axis=0)
            return np.hstack([mu, std])
    except Exception:
        pass
    
    # 如果不支持 SURF，返回空数组，不影响拼接
    return np.array([]) 

def extract_features_combined(image_path):
    """特征融合主函数：HOG + LBP + SIFT + SURF"""
    img = cv2.imread(image_path)
    if img is None: return None
    
    feat_list = []
    
    # 1. HOG
    feat_list.append(get_hog_features(img))
    
    # 2. LBP (BLP)
    feat_list.append(get_lbp_features(img))
    
    # 3. SIFT
    feat_list.append(get_sift_features(img))
    
    # 4. SURF (如果可用)
    # surf_feat = get_surf_features(img)
    # if surf_feat.size > 0:
    #     feat_list.append(surf_feat)
    
    # 拼接所有特征
    return np.hstack(feat_list)

# 2. 数据加载与预处理

def load_dataset_features(directory, dataset_name="Train"):
    X, y = [], []
    classes = sorted(os.listdir(directory))
    print(f"\n[{dataset_name}] 开始提取特征 (HOG+LBP+SIFT)")
    
    start_time = time.time()
    for label_name in classes:
        class_dir = os.path.join(directory, label_name)
        if not os.path.isdir(class_dir): continue
        
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # 调试开关：如果要快点跑通，取消下面这行注释
        # img_files = img_files[:20] 
        
        for img_file in img_files:
            feat = extract_features_combined(os.path.join(class_dir, img_file))
            if feat is not None:
                X.append(feat)
                y.append(label_name)
                
    print(f"[{dataset_name}] 完成。耗时: {time.time()-start_time:.2f}s, 样本数: {len(X)}")
    return np.array(X), np.array(y)

# 执行加载
print("步骤 1: 特征工程 (SIFT-HOG-LBP)")
X_train, y_train_labels = load_dataset_features(TRAIN_DIR, "训练集")
X_test, y_test_labels = load_dataset_features(TEST_DIR, "测试集")

print(f"融合后的特征向量维度: {X_train.shape[1]}")

print("步骤 2: 数据标准化")
le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)
y_test = le.transform(y_test_labels)

# 必须标准化，因为 LBP 是直方图(0-1)，SIFT 是描述子(大数值)，量纲差异巨大
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 多模型训练
print("步骤 3: 多模型训练与对比")

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
    print(f"正在训练: {name}", end="")
    t0 = time.time()
    try:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cost_time = time.time() - t0
        
        print(f"完成。Acc: {acc:.2%} | Time: {cost_time:.2f}s")
        
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

# 4. 自动可视化
print("步骤 4: 生成可视化图表")

# 图表1: 排行榜
plt.figure(figsize=(12, 7))
ax = sns.barplot(x="Accuracy", y="Model", data=results_df, palette="viridis")
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f%%', padding=3)
plt.title("融合特征 (SIFT+HOG+LBP) 模型准确率排行", fontsize=16)
plt.xlabel("准确率 (%)")
plt.tight_layout()
plt.savefig("chart_1_features_rank.png", dpi=150)

# 图表2: 最佳混淆矩阵
best_model_name = results_df.iloc[0]["Model"]
print(f"冠军模型: {best_model_name}")

plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_test, predictions[best_model_name])
sns.heatmap(cm, annot=False, cmap='Blues', fmt='d', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'最佳模型混淆矩阵 ({best_model_name})', fontsize=18)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("chart_2_confusion_matrix_features.png", dpi=150)

# 保存
with open('best_feature_model.pkl', 'wb') as f:
    pickle.dump(trained_models[best_model_name], f)

print("全部完成.新图片已保存：")
print("1. chart_1_features_rank.png")
print("2. chart_2_confusion_matrix_features.png")