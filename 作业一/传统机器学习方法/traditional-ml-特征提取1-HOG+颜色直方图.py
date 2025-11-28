# 作业一：基于PlantDoc数据集的植物病害识别与分类任务 (传统机器学习部分)
# 特征提取1 HOG+颜色直方图

import os
import cv2
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from skimage.feature import hog

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

# 配置路径
TRAIN_DIR = r"D:\PlantDoc\TRAIN"
TEST_DIR = r"D:\PlantDoc\TEST"

# 1. 特征提取函数
def extract_features(image_path, resize_shape=(128, 128)):
    img = cv2.imread(image_path)
    if img is None: return None
    try:
        img = cv2.resize(img, resize_shape)
        hist_features = []
        for i in range(3): 
            hist = cv2.calcHist([img], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                           cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        return np.hstack([hist_features, hog_features])
    except Exception as e:
        print(f"Error: {e}")
        return None

def load_dataset_features(directory, dataset_name="Train"):
    X, y = [], []
    classes = sorted(os.listdir(directory))
    print(f"\n[{dataset_name}] 开始加载数据，共 {len(classes)} 类")
    start_time = time.time()
    for label_name in classes:
        class_dir = os.path.join(directory, label_name)
        if not os.path.isdir(class_dir): continue
        img_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png'))]
        # img_files = img_files[:30] # 调试用
        for img_file in img_files:
            feat = extract_features(os.path.join(class_dir, img_file))
            if feat is not None:
                X.append(feat)
                y.append(label_name)
    print(f"[{dataset_name}] 完成，耗时: {time.time()-start_time:.2f}s, 样本数: {len(X)}")
    return np.array(X), np.array(y)


# 2. 主流程

print("步骤 1: 数据加载")
X_train, y_train_labels = load_dataset_features(TRAIN_DIR, "训练集")
X_test, y_test_labels = load_dataset_features(TEST_DIR, "测试集")

print("步骤 2: 预处理")
le = LabelEncoder()
y_train = le.fit_transform(y_train_labels)
y_test = le.transform(y_test_labels)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("步骤 3: 多模型训练")

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
    print(f"正在训练: {name}  ", end="")
    t0 = time.time()
    try:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"完成。Accuracy: {acc:.2%} (耗时: {time.time()-t0:.2f}s)")
        results.append({"Model": name, "Accuracy": acc})
        trained_models[name] = clf
        predictions[name] = y_pred
    except Exception as e:
        print(f"失败: {e}")

# 展示排行榜
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
print("\n" + "="*50)
print("传统机器学习模型排行榜")
print("="*50)
print(results_df)

print("步骤 4: 最佳模型可视化")
best_model_name = results_df.iloc[0]["Model"]
print(f"冠军模型: {best_model_name}")

best_y_pred = predictions[best_model_name]

plt.figure(figsize=(16, 14))
cm = confusion_matrix(y_test, best_y_pred)
sns.heatmap(cm, annot=False, cmap='Blues', fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=20)
plt.ylabel('True Label', fontsize=15)
plt.xlabel('Predicted Label', fontsize=15)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('confusion_matrix_final.png')
print(f"混淆矩阵已保存: confusion_matrix_final.png")

print(f"{best_model_name} 详细报告")
print(classification_report(y_test, best_y_pred, target_names=le.classes_))

# 保存模型
with open('best_model.pkl', 'wb') as f:
    pickle.dump(trained_models[best_model_name], f)

print("全部完成")