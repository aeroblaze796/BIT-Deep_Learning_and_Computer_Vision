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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 权重初始化函数
def initialize_weights(model):
    """
    对模型的所有卷积层和全连接层应用 Kaiming He 初始化。
    """
    print("Applying Kaiming He initialization")
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    return model

# 数据预处理与加载函数
def get_dataloaders(batch_size):
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
    base_dir = '.'
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(base_dir, 'TRAIN'), data_transforms['train']),
        'test': datasets.ImageFolder(os.path.join(base_dir, 'TEST'), data_transforms['test'])
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8)
        for x in ['train', 'test']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    return dataloaders, dataset_sizes

# 训练与验证循环函数
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs, patience):
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0
    final_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(num_epochs):
        final_epoch = epoch + 1
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'test']:
            if phase == 'train': model.train()
            else: model.eval()
            running_loss = 0.0
            running_corrects = 0
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
                    best_acc, best_model_wts = epoch_acc, copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    print(f'New best model found! Acc: {best_acc:.4f}')
                else:
                    patience_counter += 1
        if patience_counter >= patience:
            print(f"\n提前停止：验证集准确率连续 {patience} 个 epoch 未提升。")
            break
        print()
    time_elapsed = time.time() - start_time
    print(f'训练耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证集准确率: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model, history, best_acc, final_epoch, time_elapsed


# 结果可视化与保存函数
def save_and_plot_results(history, best_model, exp_config):
    results_dir = exp_config['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    model_name = "best_model.pth"
    model_save_path = os.path.join(results_dir, model_name)
    torch.save(best_model.state_dict(), model_save_path)
    print(f"最佳模型已保存至: {model_save_path}")

    acc = [h for h in history['train_acc']]
    val_acc = [h for h in history['test_acc']]
    loss = history['train_loss']
    val_loss = history['test_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='train_acc')
    plt.plot(epochs_range, val_acc, label='test_acc')
    plt.legend(loc='lower right')
    plt.title('acc curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='train loss')
    plt.plot(epochs_range, val_loss, label='test loss')
    plt.legend(loc='upper right')
    plt.title('loss curve')
    
    plt.suptitle(f"experiment: {exp_config['name']} training history")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_save_path = os.path.join(results_dir, 'training_history.png')
    plt.savefig(plot_save_path)
    print(f"训练历史曲线图已保存至: {plot_save_path}\n")


# 主实验运行函数
def run_experiment(exp_config):
    print(f"开始实验: {exp_config['name']}")

    dataloaders, dataset_sizes = get_dataloaders(exp_config['batch_size'])

    model = models.vgg16_bn(pretrained=exp_config['pretrained'])
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, exp_config['num_classes'])
    
    # 根据配置决定是否应用 Kaiming 初始化
    # 只有在 pretrained=False 且 init_method='kaiming' 时才执行
    if not exp_config['pretrained'] and exp_config.get('init_method') == 'kaiming':
        model = initialize_weights(model)
    elif not exp_config['pretrained']:
        print("Using default random initialization")
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=exp_config['lr'])

    best_model, history, best_acc, final_epoch, time_elapsed = train_model(
        model, criterion, optimizer, dataloaders, dataset_sizes, 
        exp_config['num_epochs'], exp_config['patience']
    )

    save_and_plot_results(history, best_model, exp_config)

    best_epoch_idx = history['test_acc'].index(best_acc.item())
    best_train_acc = history['train_acc'][best_epoch_idx]
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)

    return {
        "实验名称": exp_config['name'],
        "最佳测试集Acc": f"{best_acc.item()*100:.2f}%",
        "对应训练集Acc": f"{best_train_acc*100:.2f}%",
        "停止Epoch": final_epoch,
        "训练时间": f"{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s",
        "参数量(M)": f"{total_params / 1e6:.2f}",
        "计算量(GFLOPs)": f"{flops / 1e9:.2f}"
    }

# 脚本主入口
if __name__ == '__main__':
    experiments = [
        {
            "name": "VGG16_BN (random init)",
            "pretrained": False,
            "lr": 1e-3, # 从头训练使用大学习率
            "results_dir": "vgg16_bn_scratch_results"
        },
        {
            "name": "VGG16_BN (Kaiming init)",
            "pretrained": False,
            "init_method": "kaiming", # 标识符
            "lr": 1e-3, # 同样是从头训练，使用相同的大学习率以作公平对比
            "results_dir": "vgg16_bn_kaiming_results"
        },
        {
            "name": "VGG16_BN (use pretrained weights)",
            "pretrained": True,
            "lr": 1e-4, # 微调使用小学习率
            "results_dir": "vgg16_bn_finetune_results"
        }
    ]

    # 通用配置
    common_config = {
        "num_classes": 27,
        "batch_size": 32,
        "num_epochs": 200,
        "patience": 20
    }

    summary_results = []

    for exp in experiments:
        current_config = {**common_config, **exp}
        result_metrics = run_experiment(current_config)
        summary_results.append(result_metrics)

    print("\n" + "="*80)
    print("所有实验总结报告")
    print("="*80)
    
    summary_df = pd.DataFrame(summary_results)
    print(summary_df.to_string(index=False))
    print("="*80)