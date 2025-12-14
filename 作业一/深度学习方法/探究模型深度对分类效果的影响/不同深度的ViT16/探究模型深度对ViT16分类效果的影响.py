import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from thop import profile
import timm
import pandas as pd

import os
import time
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. 全局配置
class GlobalConfig:
    DATA_DIR = '.'
    TRAIN_DIR = os.path.join(DATA_DIR, 'TRAIN')
    TEST_DIR = os.path.join(DATA_DIR, 'TEST')
    
    NUM_CLASSES = 27
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    
    # ViT微调的通用超参数
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    
    EARLY_STOPPING_PATIENCE = 20
    
    # 结果保存的总目录
    MAIN_RESULTS_DIR = '探究模型深度对ViT16分类效果的影响'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. 数据加载函数
def get_dataloaders(config):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {
        'train': datasets.ImageFolder(config.TRAIN_DIR, data_transforms['train']),
        'test': datasets.ImageFolder(config.TEST_DIR, data_transforms['test'])
    }
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8) for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    return dataloaders, dataset_sizes

# 3. 训练循环
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, config):
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter, final_epoch = 0, 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    for epoch in range(config.NUM_EPOCHS):
        final_epoch = epoch + 1; print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}\n' + '-'*10)
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
                    patience_counter = 0
                    print(f'New best model found! Acc: {best_acc:.4f}')
                else: 
                    patience_counter += 1
        if patience_counter >= config.EARLY_STOPPING_PATIENCE: 
            print(f"\nEarly stopping: No improvement for {patience_counter} epochs.")
            break
        print()
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model, history, best_acc, final_epoch, time_elapsed

# 4. 结果保存与绘图
def save_and_plot_results(history, best_model, global_config, model_config):
    results_dir = os.path.join(global_config.MAIN_RESULTS_DIR, model_config['results_dir'])
    os.makedirs(results_dir, exist_ok=True)
    
    model_save_path = os.path.join(results_dir, f"best_{model_config['model_name']}.pth")
    torch.save(best_model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    acc, val_acc, loss, val_loss = history['train_acc'], history['test_acc'], history['train_loss'], history['test_loss']
    epochs_range = range(len(acc))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Acc')
    plt.plot(epochs_range, val_acc, label='Validation Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.suptitle(f"{model_config['display_name']} Training History")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_save_path = os.path.join(results_dir, f"training_history_{model_config['model_name']}.png")
    plt.savefig(plot_save_path)
    print(f"Training history plot saved to {plot_save_path}")
    plt.close()

# 5. 单次实验的主体函数
def run_experiment(global_config, model_config):
    print("\n" + "="*50)
    print(f"  Starting Experiment for: {model_config['display_name']}")
    print("="*50)
    
    dataloaders, dataset_sizes = get_dataloaders(global_config)
    
    model = timm.create_model(
        model_config['model_name'], 
        pretrained=False, 
        num_classes=global_config.NUM_CLASSES
    )
    
    try:
        print(f"Loading weights from: {model_config['pretrained_path']}")
        local_state_dict = torch.load(model_config['pretrained_path'], map_location=device)
        if 'head.weight' in local_state_dict: 
            del local_state_dict['head.weight']
        if 'head.bias' in local_state_dict: 
            del local_state_dict['head.bias']
        model.load_state_dict(local_state_dict, strict=False)
        print("Backbone weights loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Weight file not found at {model_config['pretrained_path']}. Skipping this model.")
        return None
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=global_config.LEARNING_RATE, weight_decay=global_config.WEIGHT_DECAY)
    
    best_model, history, best_acc, final_epoch, time_elapsed = train_model(
        model, criterion, optimizer, dataloaders, dataset_sizes, global_config
    )
    
    save_and_plot_results(history, best_model, global_config, model_config)
    
    best_epoch_idx = history['test_acc'].index(best_acc.item())
    best_train_acc = history['train_acc'][best_epoch_idx]
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    
    result_stats = {
        'Model': model_config['display_name'],
        'Test Acc (%)': f"{best_acc.item() * 100:.2f}",
        'Train Acc (%)': f"{best_train_acc * 100:.2f}",
        'Params (M)': f"{total_params / 1e6:.2f}",
        'GFLOPs': f"{flops / 1e9:.2f}",
        'Stop Epoch': final_epoch,
        'Time (min)': f"{time_elapsed / 60:.2f}"
    }
    return result_stats

# 6. 生成最终对比表格
def generate_summary_table(results_list):
    if not results_list:
        print("\nNo experiments were successfully completed.")
        return
        
    df = pd.DataFrame(results_list)
    print("\n\n" + "="*80)
    print(" " * 25 + "ViT Models Comparison Summary")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


# 7. 主执行流程
if __name__ == '__main__':
    global_config = GlobalConfig()

    # 定义所有要运行的实验
    experiment_configs = [
        {
            'display_name': 'ViT-Tiny-16',
            'model_name': 'vit_tiny_patch16_224',
            'pretrained_path': 'vit_tiny_patch16_224.pth',
            'results_dir': 'vit_tiny_results'
        },
        {
            'display_name': 'ViT-Small-16',
            'model_name': 'vit_small_patch16_224',
            'pretrained_path': 'vit_small_patch16_224.pth',
            'results_dir': 'vit_small_results'
        },
        {
            'display_name': 'ViT-Base-16',
            'model_name': 'vit_base_patch16_224',
            'pretrained_path': 'vit_base_patch16_224.pth',
            'results_dir': 'vit_base_results'
        }
    ]
    
    all_results = []
    
    # 循环执行所有实验
    for model_config in experiment_configs:
        result = run_experiment(global_config, model_config)
        if result:
            all_results.append(result)
            
    # 生成并打印最终的对比表格
    generate_summary_table(all_results)