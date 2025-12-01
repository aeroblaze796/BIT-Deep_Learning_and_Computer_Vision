import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import pandas as pd # <-- 导入 pandas

import os
import time
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary

# --- 1. 静态配置 (不会在实验中改变的参数) ---
class StaticConfig:
    DATA_DIR = '.'
    TRAIN_DIR = os.path.join(DATA_DIR, 'TRAIN')
    TEST_DIR = os.path.join(DATA_DIR, 'TEST')
    NUM_CLASSES = 27
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 20
    # 主结果文件夹，所有实验的子文件夹都将创建在这里
    PARENT_RESULTS_DIR = '探究不同超参数对DenseNet121分类效果的影响_结果'

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. 定义你要进行的实验 ---
# 这是一个配置列表，每个字典代表一次独立的实验
EXPERIMENT_CONFIGS = [
    {'lr': 0.001,  'optimizer': 'Adam'},
    {'lr': 0.0001, 'optimizer': 'Adam'},
    {'lr': 1e-5,   'optimizer': 'Adam'},
    {'lr': 0.001,  'optimizer': 'SGD'},
    {'lr': 0.0001, 'optimizer': 'SGD'},
]

# --- 3. 数据加载函数 (保持不变) ---
def get_dataloaders(config):
    # ... (这部分代码与之前完全相同，为了简洁此处省略)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {
        'train': datasets.ImageFolder(config.TRAIN_DIR, data_transforms['train']),
        'test': datasets.ImageFolder(config.TEST_DIR, data_transforms['test'])
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8)
        for x in ['train', 'test']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    return dataloaders, dataset_sizes

# --- 4. 训练函数 (保持不变) ---
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, config):
    # ... (这部分代码与之前完全相同，为了简洁此处省略)
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(config.NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss, running_corrects = 0.0, 0
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} Phase"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'test':
                if epoch_acc > best_acc:
                    best_acc, best_epoch = epoch_acc, epoch + 1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    best_train_acc = history['train_acc'][-1]
                    print(f'New best model found! Acc: {best_acc:.4f}')
                else:
                    patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered...")
            stopped_epoch = epoch + 1
            break
        print()
    
    if 'stopped_epoch' not in locals():
        stopped_epoch = config.NUM_EPOCHS

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} at epoch {best_epoch}')
    model.load_state_dict(best_model_wts)
    return model, history, best_acc.item(), best_train_acc, best_epoch, stopped_epoch, time_elapsed

# --- 5. 结果保存与绘图函数 (修改为接受动态路径) ---
def save_and_plot_results(history, best_model, exp_results_dir, exp_name):
    os.makedirs(exp_results_dir, exist_ok=True)
    
    model_save_path = os.path.join(exp_results_dir, f'best_model_{exp_name}.pth')
    torch.save(best_model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    # 绘图逻辑
    acc = history['train_acc']
    val_acc = history['test_acc']
    loss = history['train_loss']
    val_loss = history['test_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Test Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Test Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    
    plt.suptitle(f'Training History for {exp_name}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_save_path = os.path.join(exp_results_dir, f'history_{exp_name}.png')
    plt.savefig(plot_save_path)
    print(f"Plot saved to {plot_save_path}")
    plt.close() # <-- 关闭图像，防止在循环中连续显示

# --- 6. 主自动化脚本 ---
if __name__ == '__main__':
    config = StaticConfig()
    dataloaders, dataset_sizes = get_dataloaders(config)
    
    all_results = [] # 用于存储所有实验的结果

    # 创建父结果目录
    os.makedirs(config.PARENT_RESULTS_DIR, exist_ok=True)

    for i, exp_config in enumerate(EXPERIMENT_CONFIGS):
        lr = exp_config['lr']
        opt_name = exp_config['optimizer']
        exp_name = f"lr_{lr}_opt_{opt_name}"
        exp_results_dir = os.path.join(config.PARENT_RESULTS_DIR, exp_name)

        print("\n" + "="*50)
        print(f"  Running Experiment {i+1}/{len(EXPERIMENT_CONFIGS)}: {exp_name}")
        print("="*50)

        # 每次实验都重新加载预训练模型，保证公平性
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, config.NUM_CLASSES)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        
        # 根据配置选择优化器
        if opt_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif opt_name == 'SGD':
            # SGD通常需要配合动量(momentum)才能取得好效果
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Optimizer {opt_name} not supported.")

        # 训练模型
        results = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, config)
        best_model, history, best_val_acc, best_train_acc, best_epoch, stopped_epoch, total_time = results

        # 保存该次实验的结果
        save_and_plot_results(history, best_model, exp_results_dir, exp_name)

        # 收集本次实验的关键指标
        all_results.append({
            'Learning Rate': lr,
            'Optimizer': opt_name,
            'Best Val Acc': f"{best_val_acc:.4f}",
            'Train Acc @ Best': f"{best_train_acc:.4f}",
            'Best Epoch': best_epoch,
            'Stopped Epoch': stopped_epoch,
            'Training Time (s)': f"{total_time:.2f}",
        })
    
    # --- 7. 所有实验结束后，打印总结表格 ---
    print("\n\n" + "="*80)
    print("                      HYPERPARAMETER TUNING SUMMARY")
    print("="*80)
    
    if all_results:
        summary_df = pd.DataFrame(all_results)
        # 按最佳验证准确率降序排序
        summary_df = summary_df.sort_values(by='Best Val Acc', ascending=False)
        # 使用 to_string() 来保证所有列都被打印出来
        print(summary_df.to_string(index=False))
    else:
        print("No experiments were run.")
    print("="*80)