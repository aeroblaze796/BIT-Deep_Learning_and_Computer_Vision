import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from thop import profile
import pandas as pd

import os
import time
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. 全局配置
class Config:
    DATA_DIR = '.'
    TRAIN_DIR = os.path.join(DATA_DIR, 'TRAIN')
    TEST_DIR = os.path.join(DATA_DIR, 'TEST')
    
    BASE_RESULTS_DIR = '探究数据增强对ResNet34分类效果的影响_结果'
    
    NUM_CLASSES = 27
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 15

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 2. 定义数据增强策略
def get_augmentation_pipelines():
    """定义三种强度的数据增强方案。"""
    # 所有实验的测试集变换都是相同的
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    pipelines = {
        'gentle': {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': test_transform,
            'name_cn': 'gentle'
        },
        'moderate': {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': test_transform,
            'name_cn': 'moderate'
        },
        'aggressive': {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': test_transform,
            'name_cn': 'aggressive'
        }
    }
    return pipelines

# 3. 参数化的数据加载器
def get_dataloaders(config, data_transforms):
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

# 4. 训练循环
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, config):
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0
    final_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(config.NUM_EPOCHS):
        final_epoch = epoch + 1
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train': model.train()
            else: model.eval()
            running_loss, running_corrects = 0.0, 0
            phase_cn = '训练' if phase == 'train' else '测试'
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase_cn}阶段"):
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
            print(f'{phase_cn}损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f}')
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            if phase == 'test':
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    print(f'发现新最佳模型。准确率: {best_acc:.4f}')
                else:
                    patience_counter += 1
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n已触发提前停止。")
            break
        print()

    time_elapsed = time.time() - start_time
    model.load_state_dict(best_model_wts)
    
    # 找到最佳测试准确率对应的训练准确率
    best_epoch_idx = history['test_acc'].index(best_acc.item())
    best_train_acc = history['train_acc'][best_epoch_idx]
    
    return model, best_acc, best_train_acc, final_epoch, time_elapsed, history

# 5. 参数化的结果保存器
def save_and_plot_results(history, best_model, config, strategy_name, strategy_name_cn):
    strategy_dir = os.path.join(config.BASE_RESULTS_DIR, strategy_name)
    os.makedirs(strategy_dir, exist_ok=True)
    
    model_save_path = os.path.join(strategy_dir, f'best_model_{strategy_name}.pth')
    torch.save(best_model.state_dict(), model_save_path)
    print(f"'{strategy_name_cn}' 策略的最佳模型已保存至 {model_save_path}")

    acc, val_acc = history['train_acc'], history['test_acc']
    loss, val_loss = history['train_loss'], history['test_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    #plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='train_acc')
    plt.plot(epochs_range, val_acc, label='test_acc')
    plt.legend(loc='lower right'); plt.title('acc')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='train_loss')
    plt.plot(epochs_range, val_loss, label='test_loss')
    plt.legend(loc='upper right'); plt.title('loss')
    
    plt.suptitle(f'train history (strategy: {strategy_name_cn})')
    plot_save_path = os.path.join(strategy_dir, f'history_{strategy_name}.png')
    plt.savefig(plot_save_path)
    plt.close()
    print(f"'{strategy_name_cn}' 策略的训练历史图已保存至 {plot_save_path}")

# 6. 总结表格生成器
def print_summary_table(results_list):
    df = pd.DataFrame(results_list)
    df = df[['增强策略', '最佳测试准确率 (%)', '对应训练准确率 (%)', '停止Epoch', '训练耗时 (分钟)']]
    
    print("\n" + "="*80)
    print(" " * 25 + "数据增强实验总结")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

# 7. 主执行脚本
if __name__ == '__main__':
    config = Config()
    augmentation_pipelines = get_augmentation_pipelines()
    
    all_results = []

    for strategy_name, details in augmentation_pipelines.items():
        data_transforms = details
        strategy_name_cn = details['name_cn']
        
        print(f"开始数据增强策略实验: {strategy_name_cn.upper()}")

        dataloaders, dataset_sizes = get_dataloaders(config, data_transforms)

        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

        best_model, best_test_acc, best_train_acc, final_epoch, time_elapsed, history = train_model(
            model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, config
        )
        
        save_and_plot_results(history, best_model, config, strategy_name, strategy_name_cn)

        result = {
            '增强策略': strategy_name_cn,
            '最佳测试准确率 (%)': round(best_test_acc.item() * 100, 2),
            '对应训练准确率 (%)': round(best_train_acc * 100, 2),
            '停止Epoch': final_epoch,
            '训练耗时 (分钟)': f"{time_elapsed / 60:.1f}"
        }
        all_results.append(result)
        
        print(f"\n'{strategy_name_cn.upper()}' 策略的实验已完成")

    print_summary_table(all_results)