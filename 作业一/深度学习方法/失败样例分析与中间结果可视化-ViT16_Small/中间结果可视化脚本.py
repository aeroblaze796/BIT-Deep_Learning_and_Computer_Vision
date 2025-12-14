import torch
import timm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import cv2
import types 

MODEL_PATH = 'best_vit_small_patch16_224.pth'
IMAGE_PATH = 'test_pic2.jpg'
NUM_CLASSES = 27
MODEL_NAME = 'vit_small_patch16_224'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def new_attention_forward(self, x, attn_mask=None): 
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    self.saved_attention = attn
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def visualize_attention(model_path, image_path, model_name, num_classes):
    print("Loading model")
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    last_block_attn = model.blocks[-1].attn
    last_block_attn.forward = types.MethodType(new_attention_forward, last_block_attn)
    print("Attention module monkey-patched successfully.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Image not found at {image_path}")
        return

    img_tensor = transform(img).unsqueeze(0).to(device)

    print("Performing forward pass")
    with torch.no_grad():
        output = model(img_tensor)

    att_mat = last_block_attn.saved_attention.cpu().squeeze(0)
    
    cls_att_map = att_mat[:, 0, 1:].mean(dim=0)
    
    grid_size = int(np.sqrt(cls_att_map.shape[0]))
    attention_grid = cls_att_map.reshape(grid_size, grid_size).detach().numpy()
    
    mask = cv2.resize(attention_grid, (img.width, img.height))
    
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))
    fig.suptitle('ViT-Small-16 Attention Visualization', fontsize=16)

    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    im = ax2.imshow(mask, cmap='viridis')
    ax2.set_title('Attention Heatmap')
    ax2.axis('off')
    fig.colorbar(im, ax=ax2)
    ax3.imshow(img)
    ax3.imshow(mask, cmap='viridis', alpha=0.5)
    ax3.set_title('Overlay')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()
    
    save_path = 'attention_visualization.png'
    fig.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == '__main__':
    visualize_attention(MODEL_PATH, IMAGE_PATH, MODEL_NAME, NUM_CLASSES)