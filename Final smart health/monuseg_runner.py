#!/usr/bin/env python3
"""
MoNuSeg数据集处理脚本
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from datetime import datetime
import glob
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加当前目录到Python路径
sys.path.append('.')

def parse_xml_annotation(xml_path):
    """解析XML标注文件，提取细胞核坐标"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    nuclei = []
    for annotation in root.findall('.//Annotation'):
        for region in annotation.findall('.//Region'):
            vertices = []
            for vertex in region.findall('.//Vertex'):
                x = float(vertex.get('X'))
                y = float(vertex.get('Y'))
                vertices.append([x, y])
            
            if len(vertices) > 2:  # 至少需要3个点形成多边形
                nuclei.append(np.array(vertices))
    
    return nuclei

def create_mask_from_nuclei(nuclei, image_shape):
    """从细胞核坐标创建掩码"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    for nucleus in nuclei:
        # 将坐标转换为整数
        nucleus_int = nucleus.astype(np.int32)
        # 填充多边形
        cv2.fillPoly(mask, [nucleus_int], 255)
    
    return mask

def load_monuseg_data(data_dir, is_training=True):
    """加载MoNuSeg数据"""
    print(f"加载{'训练' if is_training else '测试'}数据...")
    
    if is_training:
        images_dir = os.path.join(data_dir, "MoNuSeg 2018 Training Data", "Tissue Images")
        annotations_dir = os.path.join(data_dir, "MoNuSeg 2018 Training Data", "Annotations")
    else:
        # 测试数据在同一目录下
        images_dir = data_dir
        annotations_dir = data_dir
    
    # 获取所有图像文件
    image_files = glob.glob(os.path.join(images_dir, "*.tif"))
    
    data_pairs = []
    
    for img_path in tqdm(image_files, desc="处理数据"):
        # 获取对应的XML文件
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(annotations_dir, f"{base_name}.xml")
        
        if os.path.exists(xml_path):
            # 读取图像
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # 转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 解析XML标注
            nuclei = parse_xml_annotation(xml_path)
            
            # 创建掩码
            mask = create_mask_from_nuclei(nuclei, image.shape)
            
            data_pairs.append({
                'image': image,
                'mask': mask,
                'image_path': img_path,
                'xml_path': xml_path
            })
    
    print(f"成功加载 {len(data_pairs)} 个样本")
    return data_pairs

def preprocess_data(data_pairs, target_size=512):
    """预处理数据"""
    print("预处理数据...")
    
    processed_data = []
    
    for pair in tqdm(data_pairs, desc="预处理"):
        image = pair['image']
        mask = pair['mask']
        
        # 调整图像尺寸
        image_resized = cv2.resize(image, (target_size, target_size))
        mask_resized = cv2.resize(mask, (target_size, target_size))
        
        # 标准化图像
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # 二值化掩码
        mask_binary = (mask_resized > 127).astype(np.uint8) * 255
        
        processed_data.append({
            'image': image_normalized,
            'mask': mask_binary,
            'original_image': image_resized
        })
    
    return processed_data

def create_data_loaders(processed_data, train_ratio=0.8):
    """创建数据加载器"""
    # 划分训练集和验证集
    n_samples = len(processed_data)
    n_train = int(n_samples * train_ratio)
    
    train_data = processed_data[:n_train]
    val_data = processed_data[n_train:]
    
    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    
    return train_data, val_data

class MoNuSegTransUNet(nn.Module):
    """针对MoNuSeg优化的TransUNet模型"""
    def __init__(self, img_size=512, in_channels=3, out_channels=1):
        super().__init__()
        
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 2, stride=2),  # 256+256=512
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, stride=2),   # 128+128=256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 2, stride=2),   # 64+64=128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 最终输出层
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        input_size = x.size()[2:]
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        dec4 = self.dec4(enc4)
        # 对齐enc3尺寸
        if dec4.shape[2:] != enc3.shape[2:]:
            dec4 = F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec3 = self.dec3(dec4)
        if dec3.shape[2:] != enc2.shape[2:]:
            dec3 = F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec2 = self.dec2(dec3)
        if dec2.shape[2:] != enc1.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec1 = self.dec1(dec2)
        # 输出resize到输入尺寸
        if dec1.shape[2:] != input_size:
            dec1 = F.interpolate(dec1, size=input_size, mode='bilinear', align_corners=False)
        output = self.final_conv(dec1)
        return output

def dice_loss(pred, target, smooth=1e-6):
    """Dice损失函数"""
    pred = torch.sigmoid(pred)
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def dice_coefficient(pred, target, smooth=1e-6):
    """计算Dice系数"""
    pred = torch.sigmoid(pred) > 0.5
    
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def train_model(model, train_data, val_data, num_epochs=20, batch_size=2):
    """训练模型"""
    print("开始训练模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = model.to(device)
    criterion = dice_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    train_losses = []
    val_losses = []
    best_dice = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 随机选择训练样本
        indices = np.random.choice(len(train_data), min(batch_size, len(train_data)), replace=False)
        
        for idx in indices:
            sample = train_data[idx]
            
            # 准备数据
            image = torch.from_numpy(sample['image'].transpose(2, 0, 1)).unsqueeze(0).to(device)
            mask = torch.from_numpy(sample['mask'].astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, mask)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(indices)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for sample in val_data[:5]:  # 只验证前5个样本
                image = torch.from_numpy(sample['image'].transpose(2, 0, 1)).unsqueeze(0).to(device)
                mask = torch.from_numpy(sample['mask'].astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
                
                output = model(image)
                loss = criterion(output, mask)
                dice = dice_coefficient(output, mask)
                
                val_loss += loss.item()
                val_dice += dice
        
        val_loss /= min(5, len(val_data))
        val_dice /= min(5, len(val_data))
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")
        print(f"  验证Dice: {val_dice:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"训练完成！最佳验证Dice: {best_dice:.4f}")
    return model, train_losses, val_losses

def evaluate_model(model, test_data):
    """评估模型"""
    print("评估模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    dice_scores = []
    
    with torch.no_grad():
        for sample in tqdm(test_data, desc="评估"):
            image = torch.from_numpy(sample['image'].transpose(2, 0, 1)).unsqueeze(0).to(device)
            mask = torch.from_numpy(sample['mask'].astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            
            output = model(image)
            dice = dice_coefficient(output, mask)
            dice_scores.append(dice)
    
    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    print(f"平均Dice系数: {avg_dice:.4f} ± {std_dice:.4f}")
    print(f"最高Dice系数: {np.max(dice_scores):.4f}")
    print(f"最低Dice系数: {np.min(dice_scores):.4f}")
    
    return dice_scores

def visualize_results(model, test_data, num_samples=3):
    """可视化结果"""
    print("生成可视化结果...")
    
    os.makedirs('results', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        if i >= len(test_data):
            break
            
        sample = test_data[i]
        
        # 原始图像
        axes[i, 0].imshow(sample['original_image'])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # 真实掩码
        axes[i, 1].imshow(sample['mask'], cmap='gray')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')
        
        # 预测掩码
        with torch.no_grad():
            image = torch.from_numpy(sample['image'].transpose(2, 0, 1)).unsqueeze(0).to(device)
            output = model(image)
            pred = torch.sigmoid(output) > 0.5
            pred_mask = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
        
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/monuseg_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("可视化结果已保存到 results/monuseg_results.png")

def main():
    """主函数"""
    print("🏥 MoNuSeg医学图像分割项目")
    print("=" * 50)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 步骤1: 加载训练数据
    train_data_pairs = load_monuseg_data("data", is_training=True)
    
    # 步骤2: 加载测试数据
    test_data_pairs = load_monuseg_data("data/MoNuSegTestData", is_training=False)
    
    # 步骤3: 预处理数据
    train_processed = preprocess_data(train_data_pairs)
    test_processed = preprocess_data(test_data_pairs)
    
    # 步骤4: 创建数据加载器
    train_data, val_data = create_data_loaders(train_processed)
    
    # 步骤5: 创建模型
    model = MoNuSegTransUNet(img_size=512, in_channels=3, out_channels=1)
    
    # 步骤6: 训练模型
    model, train_losses, val_losses = train_model(model, train_data, val_data, num_epochs=15)
    
    # 步骤7: 评估模型
    dice_scores = evaluate_model(model, test_processed)
    
    # 步骤8: 可视化结果
    visualize_results(model, test_processed)
    
    # 步骤9: 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(dice_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Test Set Dice Coefficient Distribution')
    plt.xlabel('Dice Coefficient')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*50)
    print("🎉 项目运行完成！")
    print("="*50)
    print(f"最终平均Dice系数: {np.mean(dice_scores):.4f}")
    print("结果文件:")
    print("  - results/monuseg_results.png")
    print("  - results/training_curves.png")
    print("  - best_model.pth")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main() 