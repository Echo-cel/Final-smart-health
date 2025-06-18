#!/usr/bin/env python3
"""
简化的运行脚本
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append('.')

def create_sample_data():
    """创建示例数据"""
    print("创建示例数据...")
    
    # 创建目录
    os.makedirs('data/processed/images', exist_ok=True)
    os.makedirs('data/processed/masks', exist_ok=True)
    os.makedirs('data/train/images', exist_ok=True)
    os.makedirs('data/train/masks', exist_ok=True)
    os.makedirs('data/val/images', exist_ok=True)
    os.makedirs('data/val/masks', exist_ok=True)
    os.makedirs('data/test/images', exist_ok=True)
    os.makedirs('data/test/masks', exist_ok=True)
    
    # 创建20个示例样本
    for i in range(20):
        # 创建背景图像
        img = np.random.randint(180, 220, (512, 512, 3), dtype=np.uint8)
        
        # 创建掩码
        mask = np.zeros((512, 512), dtype=np.uint8)
        
        # 添加细胞核
        for _ in range(np.random.randint(5, 15)):
            center_x = np.random.randint(50, 462)
            center_y = np.random.randint(50, 462)
            radius = np.random.randint(10, 30)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            cv2.circle(img, (center_x, center_y), radius, (100, 100, 100), -1)
        
        # 保存文件
        filename = f'sample_{i:03d}.png'
        
        # 根据索引分配到不同数据集
        if i < 14:  # 70% 训练集
            cv2.imwrite(f'data/train/images/{filename}', img)
            cv2.imwrite(f'data/train/masks/{filename}', mask)
        elif i < 18:  # 20% 验证集
            cv2.imwrite(f'data/val/images/{filename}', img)
            cv2.imwrite(f'data/val/masks/{filename}', mask)
        else:  # 10% 测试集
            cv2.imwrite(f'data/test/images/{filename}', img)
            cv2.imwrite(f'data/test/masks/{filename}', mask)
    
    print("示例数据创建完成！")

def simple_transunet():
    """简化的TransUNet模型"""
    class SimpleTransUNet(nn.Module):
        def __init__(self):
            super().__init__()
            # 简化的编码器
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            
            # 简化的解码器
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, 1)
            )
        
        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
    
    return SimpleTransUNet()

def train_model():
    """训练模型"""
    print("开始训练模型...")
    
    # 创建模型
    model = simple_transunet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建简单的数据加载器
    train_images = []
    train_masks = []
    
    # 加载训练数据
    for i in range(14):
        img_path = f'data/train/images/sample_{i:03d}.png'
        mask_path = f'data/train/masks/sample_{i:03d}.png'
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256))
            mask = (mask > 127).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            
            train_images.append(img)
            train_masks.append(mask)
    
    # 训练几个epoch
    model.train()
    for epoch in range(5):
        total_loss = 0
        for i, (img, mask) in enumerate(zip(train_images, train_masks)):
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_images):.4f}")
    
    print("训练完成！")
    return model

def evaluate_model(model):
    """评估模型"""
    print("评估模型...")
    
    model.eval()
    test_images = []
    test_masks = []
    
    # 加载测试数据
    for i in range(18, 20):
        img_path = f'data/test/images/sample_{i:03d}.png'
        mask_path = f'data/test/masks/sample_{i:03d}.png'
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (256, 256))
            mask = (mask > 127).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            
            test_images.append(img)
            test_masks.append(mask)
    
    # 计算Dice系数
    dice_scores = []
    with torch.no_grad():
        for img, mask in zip(test_images, test_masks):
            output = model(img)
            pred = torch.sigmoid(output) > 0.5
            
            intersection = (pred * mask).sum()
            union = pred.sum() + mask.sum()
            dice = (2 * intersection) / (union + 1e-6)
            dice_scores.append(dice.item())
    
    avg_dice = np.mean(dice_scores)
    print(f"平均Dice系数: {avg_dice:.4f}")
    
    return avg_dice

def visualize_results(model):
    """可视化结果"""
    print("生成可视化结果...")
    
    os.makedirs('results', exist_ok=True)
    
    # 加载一个测试样本
    img_path = 'data/test/images/sample_018.png'
    mask_path = 'data/test/masks/sample_018.png'
    
    if os.path.exists(img_path) and os.path.exists(mask_path):
        # 加载图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (256, 256))
        
        # 加载掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_resized = cv2.resize(mask, (256, 256))
        mask_binary = (mask_resized > 127).astype(np.uint8) * 255
        
        # 预测
        model.eval()
        with torch.no_grad():
            img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            output = model(img_tensor)
            pred = torch.sigmoid(output) > 0.5
            pred_mask = pred.squeeze().numpy().astype(np.uint8) * 255
        
        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_resized)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        axes[1].imshow(mask_binary, cmap='gray')
        axes[1].set_title('真实掩码')
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('预测掩码')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/sample_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("可视化结果已保存到 results/sample_results.png")

def main():
    """主函数"""
    print("🏥 智慧医疗 - 医学图像分割项目")
    print("=" * 50)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 步骤1: 创建示例数据
    create_sample_data()
    
    # 步骤2: 训练模型
    model = train_model()
    
    # 步骤3: 评估模型
    dice_score = evaluate_model(model)
    
    # 步骤4: 可视化结果
    visualize_results(model)
    
    print("\n" + "="*50)
    print("🎉 项目运行完成！")
    print("="*50)
    print(f"最终Dice系数: {dice_score:.4f}")
    print("结果文件: results/sample_results.png")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main() 