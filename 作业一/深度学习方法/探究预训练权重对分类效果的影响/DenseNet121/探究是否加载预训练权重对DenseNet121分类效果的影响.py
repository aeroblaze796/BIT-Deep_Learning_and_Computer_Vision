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
from torchinfo import summary
import pandas as pd

# 全局配置
DATA_DIR = '.'
TRAIN_DIR = os.path.join(DATA_DIR, 'TRAIN')
TEST_DIR = os.path.join(DATA_DIR, 'TEST')
PARENT_RESULTS_DIR = '探究是否加载预训练权重对DenseNet121分类效果的影响_结果'
NUM_CLASSES = 27
BATCH_SIZE = 32
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Kaiming 初始化
def kaiming_init_weights(m):
    """
    对模型的卷积层和全连接层应用 Kaiming He 初始化。
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# 数据加载
def get_dataloaders(train_dir, test_dir, batch_size):
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
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test'])
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=8)
        for x in ['train', 'test']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    return dataloaders, dataset_sizes

# 训练循环
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs, patience):
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()
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
                    best_acc, best_model_wts = epoch_acc, copy.deepcopy(model.state_dict())
                    patience_counter, best_epoch = 0, epoch + 1
                    best_train_acc = history['train_acc'][-1]
                else:
                    patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs of no improvement.")
            break
        print()
    
    stopped_epoch = epoch + 1
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} at epoch {best_epoch}')
    model.load_state_dict(best_model_wts)
    return model, history, best_acc.item(), best_train_acc, best_epoch, stopped_epoch, time_elapsed

# 结果保存与绘图
def save_and_plot_results(history, best_model, results_dir, model_name, plot_name):
    os.makedirs(results_dir, exist_ok=True)
    model_save_path = os.path.join(results_dir, model_name)
    torch.save(best_model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.suptitle(f'{model_name} Training History')
    plot_save_path = os.path.join(results_dir, plot_name)
    plt.savefig(plot_save_path)
    print(f"Plot saved to {plot_save_path}")
    plt.close()

# 打印最终对比表格
def print_summary_table(all_results):
    df = pd.DataFrame(all_results)
    df = df.set_index('Strategy')
    print("\n" + "="*80)
    print(" " * 20 + "Comparison of Weight Initialization Strategies")
    print("="*80)
    print(df.to_string())
    print("="*80)

# 主执行函数
if __name__ == '__main__':
    experiments = [
        {
            'name': 'Pre-trained',
            'use_pretrained': True,
            'init_function': None,
            'lr': 1e-4, # 较小的学习率用于微调
        },
        {
            'name': 'Kaiming Init',
            'use_pretrained': False,
            'init_function': kaiming_init_weights,
            'lr': 1e-3, # 较大的学习率用于从头训练
        },
        {
            'name': 'Random Init',
            'use_pretrained': False,
            'init_function': None, # Pytorch 默认的随机初始化
            'lr': 1e-3, # 较大的学习率用于从头训练
        },
    ]

    all_experiment_results = []
    dataloaders, dataset_sizes = get_dataloaders(TRAIN_DIR, TEST_DIR, BATCH_SIZE)

    for exp in experiments:
        print(f"  Starting Experiment: {exp['name']} Weights")
        print(f"  Learning Rate: {exp['lr']}")

        model = models.densenet121(pretrained=exp['use_pretrained'])
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)

        if not exp['use_pretrained'] and exp['init_function']:
            print(f"Applying {exp['name']} to the model")
            model.apply(exp['init_function'])
        
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=exp['lr'])
        criterion = nn.CrossEntropyLoss()

        results = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, NUM_EPOCHS, EARLY_STOPPING_PATIENCE)
        best_model, history, best_val_acc, best_train_acc, best_epoch, stopped_epoch, total_time = results

        exp_results_dir = os.path.join(PARENT_RESULTS_DIR, exp['name'].lower().replace(' ', '_'))
        save_and_plot_results(
            history, 
            best_model, 
            results_dir=exp_results_dir,
            model_name=f"best_model_{exp['name'].lower().replace(' ', '_')}.pth",
            plot_name=f"history_{exp['name'].lower().replace(' ', '_')}.png"
        )
        
        all_experiment_results.append({
            'Strategy': exp['name'],
            'Best Val Acc': f"{best_val_acc:.4f}",
            'Train Acc': f"{best_train_acc:.4f}",
            'Best Epoch': best_epoch,
            'Stopped Epoch': stopped_epoch,
            'Time (min)': f"{total_time / 60:.2f}"
        })
        time.sleep(2)

    print_summary_table(all_experiment_results)