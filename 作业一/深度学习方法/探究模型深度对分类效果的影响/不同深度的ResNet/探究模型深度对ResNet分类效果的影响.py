import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from thop import profile
import pandas as pd

import os
import time
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. 配置
class Config:
    DATA_DIR = '.'
    TRAIN_DIR = os.path.join(DATA_DIR, 'TRAIN')
    TEST_DIR = os.path.join(DATA_DIR, 'TEST')
    
    RESULTS_PARENT_DIR = '探究模型深度对ResNet分类效果的影响_结果'
    
    # 要测试的模型列表
    MODELS_TO_TEST = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    
    # 用于公平对比的共享超参数
    NUM_CLASSES = 27
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}\n")

# 2. 数据加载
def get_dataloaders(config):
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

# 3. 训练循环
def train_model(model, model_name, criterion, optimizer, dataloaders, dataset_sizes, config):
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0
    final_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(config.NUM_EPOCHS):
        final_epoch = epoch + 1
        print(f'模型: {model_name.upper()} | Epoch {epoch+1}/{config.NUM_EPOCHS}')
        
        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()}阶段")
            for inputs, labels in progress_bar:
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
            print(f'-> {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            if phase == 'test':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
        
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n提前停止机制已为 {model_name} 触发。")
            break
        print("-" * 15)

    time_elapsed = time.time() - start_time
    model.load_state_dict(best_model_wts)
    return model, history, best_acc, final_epoch, time_elapsed

# 4. 结果保存与绘图
def save_and_plot_results(history, best_model, model_name, config):
    model_results_dir = os.path.join(config.RESULTS_PARENT_DIR, f"{model_name}_results")
    os.makedirs(model_results_dir, exist_ok=True)
    
    model_save_path = os.path.join(model_results_dir, f'best_{model_name}_model.pth')
    torch.save(best_model.state_dict(), model_save_path)
    
    acc, val_acc = history['train_acc'], history['test_acc']
    loss, val_loss = history['train_loss'], history['test_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='train_acc')
    plt.plot(epochs_range, val_acc, label='test_acc')
    plt.legend(loc='lower right')
    plt.title(f'{model_name.capitalize()} acc curve')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='train_loss')
    plt.plot(epochs_range, val_loss, label='test_loss')
    plt.legend(loc='upper right')
    plt.title(f'{model_name.capitalize()} loss curve')
    
    plot_save_path = os.path.join(model_results_dir, f'history_{model_name}.png')
    plt.savefig(plot_save_path)
    plt.close()
    print(f"{model_name} 的结果已保存至 {model_results_dir}")

# 5. 模型复杂度计算
def get_model_complexity(model):
    # 计算可训练参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 计算GFLOPs
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    # 返回以百万为单位的参数量和以G为单位的FLOPs
    return total_params / 1e6, flops / 1e9

# 6. 主执行函数
if __name__ == '__main__':
    config = Config()
    os.makedirs(config.RESULTS_PARENT_DIR, exist_ok=True)
    dataloaders, dataset_sizes = get_dataloaders(config)
    results_summary = []

    for model_name in config.MODELS_TO_TEST:
        print(f"\n{'='*25} 开始训练 {model_name.upper()} {'='*25}")
        
        model = getattr(models, model_name)(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)
        model = model.to(device)

        params_M, gflops = get_model_complexity(model)
        
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        best_model, history, best_test_acc, final_epoch, time_elapsed = train_model(
            model, model_name, criterion, optimizer, dataloaders, dataset_sizes, config
        )
        
        save_and_plot_results(history, best_model, model_name, config)
        
        best_epoch_idx = history['test_acc'].index(best_test_acc.item())
        best_train_acc = history['train_acc'][best_epoch_idx]
        
        results_summary.append({
            '模型': model_name,
            '最佳测试Acc (%)': best_test_acc.item() * 100,
            '对应训练Acc (%)': best_train_acc * 100,
            '参数量 (M)': params_M,
            '计算量 (GFLOPs)': gflops,
            '停止Epoch': final_epoch,
            '训练时间 (s)': time_elapsed
        })

    # 7. 最终总结报告
    print(f"\n{'='*25} 最终对比总结 {'='*25}")
    
    summary_df = pd.DataFrame(results_summary)
    summary_df['最佳测试Acc (%)'] = summary_df['最佳测试Acc (%)'].round(2)
    summary_df['对应训练Acc (%)'] = summary_df['对应训练Acc (%)'].round(2)
    summary_df['参数量 (M)'] = summary_df['参数量 (M)'].round(2)
    summary_df['计算量 (GFLOPs)'] = summary_df['计算量 (GFLOPs)'].round(2)
    summary_df['训练时间 (s)'] = summary_df['训练时间 (s)'].round(0).astype(int)
    
    print(summary_df.to_string(index=False))

    summary_csv_path = os.path.join(config.RESULTS_PARENT_DIR, 'resnet_comparison_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n总结表格已保存至: {summary_csv_path}")