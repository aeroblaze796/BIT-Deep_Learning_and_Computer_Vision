# 在本地电脑上运行
import timm
model = timm.create_model('vit_small_patch16_224', pretrained=True)
# 自动下载权重到本地缓存
# 将名字改为tiny/small/base/large下载不同版本
import torch
torch.save(model.state_dict(), 'vit_small_patch16_224.pth')
print("ViT权重已保存为 vit_small_patch16_224.pth")