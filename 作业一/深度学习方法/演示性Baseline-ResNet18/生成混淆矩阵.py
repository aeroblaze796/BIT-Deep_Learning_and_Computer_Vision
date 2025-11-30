import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# 1. 配置
class EvalConfig:
    DATA_DIR = '../'
    TEST_DIR = os.path.join(DATA_DIR, 'TEST')
    RESULTS_DIR = './' 
    MODEL_PATH = os.path.join(RESULTS_DIR, 'best_baseline_model.pth') 
    OUTPUT_FILE = os.path.join(RESULTS_DIR, 'confusion_matrix.png')

    # 模型参数
    NUM_CLASSES = 27
    BATCH_SIZE = 32  # 评估时可以使用稍大的batch size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备是: {device}")

# 2. 加载模型
def load_model(config):
    """加载预训练模型并载入权重"""
    print(f"正在从 '{config.MODEL_PATH}' 加载模型")
    
    # 初始化与训练时相同的模型结构
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)
    
    # 加载已保存的权重
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"找不到模型权重文件 '{config.MODEL_PATH}'")
    
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model = model.to(device)
    
    model.eval() 
    
    print("模型加载成功")
    return model

# 3. 加载测试数据
def get_test_loader(config):
    """创建测试集的数据加载器"""
    # 变换操作必须与训练时的验证/测试集变换完全一致
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(config.TEST_DIR, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    class_names = test_dataset.classes
    print(f"测试集包含 {len(test_dataset)} 张图片。")
    return test_loader, class_names

# 4. 执行评估
def evaluate(model, test_loader):
    """在测试集上运行模型并收集预测结果和真实标签"""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="正在测试集上评估"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_preds, all_labels

# 5. 绘制并保存混淆矩阵
def plot_confusion_matrix(preds, labels, class_names, config):
    """计算、绘制并保存混淆矩阵"""
    print("正在生成混淆矩阵")
    
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix', fontsize=20)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    
    plt.xticks(rotation=45, ha='right') 
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    plt.savefig(config.OUTPUT_FILE)
    print(f"混淆矩阵图像已保存至: {config.OUTPUT_FILE}")
    
    plt.show()

# 主函数
if __name__ == '__main__':
    config = EvalConfig()
    model = load_model(config)
    test_loader, class_names = get_test_loader(config)
    all_predictions, all_true_labels = evaluate(model, test_loader)
    plot_confusion_matrix(all_predictions, all_true_labels, class_names, config)