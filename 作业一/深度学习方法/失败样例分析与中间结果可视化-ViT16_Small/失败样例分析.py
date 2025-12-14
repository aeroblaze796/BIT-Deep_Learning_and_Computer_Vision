import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from tqdm import tqdm

# 1. 配置
class Config:
    MODEL_PATH = 'best_vit_small_patch16_224.pth'  # 训练好的ViT-Small模型权重
    TEST_DIR = '../../TEST'                             # 测试集数据文件夹
    
    MODEL_NAME = 'vit_small_patch16_224'          # 使用的timm模型名
    NUM_CLASSES = 27                              # 任务类别数
    BATCH_SIZE = 32                               # 推理时可以使用稍大的批次大小
    
    RESULTS_FILE = 'confusion_matrix.png'         # 结果保存文件名

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 加载测试数据
def get_test_loader(config):
    """
    创建并返回测试集的DataLoader。
    这里的预处理必须和训练时的验证/测试集预处理完全一致！
    """
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=test_transform)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,  # 推理时不需要打乱顺序
        num_workers=4
    )
    
    return test_loader, test_dataset.classes

# 3. 获取所有预测结果和真实标签
def get_predictions(model, loader):
    """在测试集上运行模型并返回所有预测和标签"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing on test set"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

# 4. 绘制并保存混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """计算并绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(18, 15))
    heatmap = sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    
    plt.title('ViT-Small-16 Confusion Matrix', fontsize=20)
    plt.ylabel('True Label', fontsize=15)
    plt.xlabel('Predicted Label', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.show()

# 5. 主执行函数
if __name__ == '__main__':
    config = Config()

    test_loader, class_names = get_test_loader(config)
    print(f"Found {len(class_names)} classes.")

    print("Loading pre-trained model")
    model = timm.create_model(
        config.MODEL_NAME, 
        pretrained=False, 
        num_classes=config.NUM_CLASSES
    )
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weights not found at '{config.MODEL_PATH}'. Please check the path.")
        exit()
        
    model.to(device)
    print("Model loaded successfully.")

    true_labels, pred_labels = get_predictions(model, test_loader)

    plot_confusion_matrix(true_labels, pred_labels, class_names, config.RESULTS_FILE)