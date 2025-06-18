#!/usr/bin/env python3
"""
MoNuSeg细胞核分割预测脚本
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加当前目录到Python路径
sys.path.append('.')

class MoNuSegTransUNet(torch.nn.Module):
    """针对MoNuSeg优化的TransUNet模型"""
    def __init__(self, img_size=512, in_channels=3, out_channels=1):
        super().__init__()
        
        # 编码器
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        self.enc2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        
        self.enc3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        
        self.enc4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.dec4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 2, stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        
        self.dec3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 128, 2, stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        
        self.dec2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 64, 2, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        self.dec1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 32, 2, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True)
        )
        
        # 最终输出层
        self.final_conv = torch.nn.Conv2d(32, out_channels, kernel_size=1)

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

def load_model(model_path='best_model.pth'):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        print("请先运行训练脚本生成模型文件")
        return None
    
    # 创建模型
    model = MoNuSegTransUNet(img_size=512, in_channels=3, out_channels=1)
    
    # 加载权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"模型加载成功，使用设备: {device}")
    return model

def preprocess_image(image_path, target_size=512):
    """预处理输入图像"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return None
    
    # 转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 保存原始尺寸
    original_size = image.shape[:2]
    
    # 调整尺寸
    image_resized = cv2.resize(image, (target_size, target_size))
    
    # 标准化
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    return {
        'original': image,
        'resized': image_resized,
        'normalized': image_normalized,
        'original_size': original_size
    }

def predict_single_image(model, image_data, threshold=0.5):
    """对单张图像进行预测"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 准备输入
    image_tensor = torch.from_numpy(image_data['normalized'].transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output)
        pred_mask = (prob > threshold).float()
    
    # 转换为numpy
    pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
    
    # 调整回原始尺寸
    pred_mask_resized = cv2.resize(pred_mask, (image_data['original_size'][1], image_data['original_size'][0]))
    
    return {
        'probability': prob.squeeze().cpu().numpy(),
        'mask': pred_mask,
        'mask_resized': pred_mask_resized
    }

def visualize_prediction(image_data, prediction, save_path=None):
    """可视化预测结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image_data['original'])
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 概率图
    axes[1].imshow(prediction['probability'], cmap='hot')
    axes[1].set_title('Prediction Probability')
    axes[1].axis('off')
    
    # 分割掩码
    axes[2].imshow(prediction['mask'], cmap='gray')
    axes[2].set_title('Segmentation Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果已保存到: {save_path}")
    
    plt.show()

def predict_batch_images(model, image_dir, output_dir='predictions'):
    """批量预测图像"""
    print(f"批量预测图像目录: {image_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        print(f"在目录 {image_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    results = []
    
    for image_path in tqdm(image_files, desc="预测中"):
        try:
            # 预处理图像
            image_data = preprocess_image(image_path)
            if image_data is None:
                continue
            
            # 预测
            prediction = predict_single_image(model, image_data)
            
            # 保存结果
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(output_dir, f"{base_name}_prediction.png")
            
            # 可视化并保存
            visualize_prediction(image_data, prediction, save_path)
            
            # 保存掩码
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, prediction['mask_resized'])
            
            results.append({
                'image_path': image_path,
                'prediction_path': save_path,
                'mask_path': mask_path
            })
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
    
    print(f"批量预测完成，共处理 {len(results)} 个图像")
    return results

def main():
    """主函数"""
    print("🔬 MoNuSeg细胞核分割预测系统")
    print("=" * 50)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载模型
    model = load_model()
    if model is None:
        return
    
    print("\n请选择预测模式:")
    print("1. 单张图像预测")
    print("2. 批量图像预测")
    print("3. 使用测试数据预测")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 单张图像预测
        image_path = input("请输入图像路径: ").strip()
        if not os.path.exists(image_path):
            print("图像文件不存在!")
            return
        
        # 预处理图像
        image_data = preprocess_image(image_path)
        if image_data is None:
            return
        
        # 预测
        prediction = predict_single_image(model, image_data)
        
        # 可视化结果
        visualize_prediction(image_data, prediction, 'results/single_prediction.png')
        
        # 保存掩码
        cv2.imwrite('results/single_prediction_mask.png', prediction['mask_resized'])
        print("预测结果已保存到 results/ 目录")
        
    elif choice == "2":
        # 批量预测
        image_dir = input("请输入图像目录路径: ").strip()
        if not os.path.exists(image_dir):
            print("目录不存在!")
            return
        
        results = predict_batch_images(model, image_dir)
        
    elif choice == "3":
        # 使用测试数据预测
        test_dir = "data/MoNuSegTestData"
        if not os.path.exists(test_dir):
            print("测试数据目录不存在!")
            return
        
        print(f"使用测试数据目录: {test_dir}")
        results = predict_batch_images(model, test_dir, 'results/test_predictions')
        
    else:
        print("无效选择!")
        return
    
    print("\n" + "="*50)
    print("🎉 预测完成！")
    print("="*50)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main() 