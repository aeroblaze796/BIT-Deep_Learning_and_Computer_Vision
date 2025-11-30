import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import os
import time
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. 配置与超参数
class Config:
    DATA_DIR = '.'  # 数据集根目录
    TRAIN_DIR = os.path.join(DATA_DIR, 'TRAIN')
    TEST_DIR = os.path.join(DATA_DIR, 'TEST')
    
    # 训练参数
    NUM_CLASSES = 27
    BATCH_SIZE = 32
    NUM_EPOCHS = 100  # 设置一个较高的上限，由提前停止来控制
    LEARNING_RATE = 0.001
    
    # 提前停止参数
    EARLY_STOPPING_PATIENCE = 10
    
    # 结果保存路径
    RESULTS_DIR = '演示性Baseline-ResNet18_结果'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 2. 数据预处理与加载
def get_dataloaders(config):
    # 为训练集和测试集定义不同的变换
    # 训练集使用简单的数据增强（随机翻转）
    # 测试集不使用数据增强，只做必要的尺寸调整和归一化
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
    
    # 创建DataLoader
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
        for x in ['train', 'test']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    print(f"训练集大小: {dataset_sizes['train']}")
    print(f"测试集大小: {dataset_sizes['test']}")
    
    return dataloaders, dataset_sizes, class_names

# 3. 训练与验证循环
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, config):
    start_time = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 用于提前停止的计数器
    patience_counter = 0
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(config.NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} Phase"):
                inputs = inputs.to(device)
                labels = labels.to(device)

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
            history[f'{phase}_acc'].append(epoch_acc.item()) # .item() to get python number

            # 检查是否为最佳模型，并实现提前停止
            if phase == 'test':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0 # 重置计数器
                    print(f'New best model found. Acc: {best_acc:.4f}')
                else:
                    patience_counter += 1

        # 检查是否需要提前停止
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {config.EARLY_STOPPING_PATIENCE} epochs of no improvement.")
            break
        print()

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history


# 4. 结果可视化与保存
def save_and_plot_results(history, best_model, config):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # 1. 保存最佳模型
    model_save_path = os.path.join(config.RESULTS_DIR, 'best_baseline_model.pth')
    torch.save(best_model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    # 2. 绘制并保存曲线图
    acc = [h for h in history['train_acc']]
    val_acc = [h for h in history['test_acc']]
    loss = history['train_loss']
    val_loss = history['test_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.suptitle('Baseline Model Training History')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_save_path = os.path.join(config.RESULTS_DIR, 'training_history.png')
    plt.savefig(plot_save_path)
    print(f"Training history plot saved to {plot_save_path}")
    plt.show()


# 5. 主函数
if __name__ == '__main__':
    config = Config()

    dataloaders, dataset_sizes, class_names = get_dataloaders(config)
    
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    best_model, history = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, config)
    
    save_and_plot_results(history, best_model, config)