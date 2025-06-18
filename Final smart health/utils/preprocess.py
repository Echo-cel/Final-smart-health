import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import argparse
from utils.dataset import split_dataset

def preprocess_images(input_dir, output_dir, target_size=512):
    """
    预处理图像数据
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        target_size: 目标图像尺寸
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像
    for filename in tqdm(image_files, desc="预处理图像"):
        # 读取图像
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"警告：无法读取图像 {filename}")
            continue
        
        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸
        image = cv2.resize(image, (target_size, target_size))
        
        # 标准化到[0,1]
        image = image.astype(np.float32) / 255.0
        
        # 保存处理后的图像
        output_path = os.path.join(output_dir, 'images', filename)
        # 转换回uint8用于保存
        image_save = (image * 255).astype(np.uint8)
        cv2.imwrite(output_path, cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR))
    
    print(f"图像预处理完成，保存到 {output_dir}")

def create_sample_data(output_dir, num_samples=50, image_size=512):
    """
    创建示例数据用于演示
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
        image_size: 图像尺寸
    """
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    
    print(f"创建 {num_samples} 个示例样本...")
    
    for i in tqdm(range(num_samples), desc="创建示例数据"):
        # 创建背景图像（模拟组织背景）
        background = np.random.randint(180, 220, (image_size, image_size, 3), dtype=np.uint8)
        
        # 添加一些纹理
        noise = np.random.normal(0, 10, (image_size, image_size, 3)).astype(np.uint8)
        background = np.clip(background + noise, 0, 255)
        
        # 创建掩码
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        
        # 添加随机数量的细胞核
        num_nuclei = np.random.randint(8, 20)
        
        for _ in range(num_nuclei):
            # 随机位置
            center_x = np.random.randint(50, image_size - 50)
            center_y = np.random.randint(50, image_size - 50)
            
            # 随机大小
            radius = np.random.randint(8, 25)
            
            # 绘制细胞核
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            # 在背景图像中添加细胞核
            nucleus_color = np.random.randint(80, 150, 3)
            cv2.circle(background, (center_x, center_y), radius, nucleus_color.tolist(), -1)
            
            # 添加细胞核边界
            cv2.circle(background, (center_x, center_y), radius, (0, 0, 0), 1)
        
        # 保存图像和掩码
        image_filename = f'sample_{i:03d}.png'
        mask_filename = f'sample_{i:03d}.png'
        
        cv2.imwrite(os.path.join(output_dir, 'images', image_filename), background)
        cv2.imwrite(os.path.join(output_dir, 'masks', mask_filename), mask)
    
    print(f"示例数据创建完成，保存到 {output_dir}")

def validate_data(data_dir):
    """
    验证数据集的完整性
    
    Args:
        data_dir: 数据目录
    """
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print("错误：缺少images或masks目录")
        return False
    
    # 获取图像文件列表
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
    
    mask_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        mask_files.extend([f for f in os.listdir(masks_dir) if f.lower().endswith(ext)])
    
    print(f"图像文件数量: {len(image_files)}")
    print(f"掩码文件数量: {len(mask_files)}")
    
    # 检查文件匹配
    missing_masks = []
    for img_file in image_files:
        mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png').replace('.tif', '.png').replace('.tiff', '.png')
        if mask_file not in mask_files:
            missing_masks.append(img_file)
    
    if missing_masks:
        print(f"警告：以下图像缺少对应的掩码文件:")
        for img in missing_masks[:10]:  # 只显示前10个
            print(f"  - {img}")
        if len(missing_masks) > 10:
            print(f"  ... 还有 {len(missing_masks) - 10} 个文件")
        return False
    
    print("数据验证通过！")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据预处理工具')
    parser.add_argument('--input_dir', type=str, help='输入数据目录')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='输出数据目录')
    parser.add_argument('--target_size', type=int, default=512, help='目标图像尺寸')
    parser.add_argument('--create_samples', action='store_true', help='创建示例数据')
    parser.add_argument('--num_samples', type=int, default=50, help='示例数据数量')
    parser.add_argument('--validate', action='store_true', help='验证数据完整性')
    parser.add_argument('--split', action='store_true', help='划分数据集')
    
    args = parser.parse_args()
    
    if args.create_samples:
        # 创建示例数据
        create_sample_data(args.output_dir, args.num_samples, args.target_size)
    elif args.input_dir:
        # 预处理真实数据
        preprocess_images(args.input_dir, args.output_dir, args.target_size)
    else:
        print("请指定 --input_dir 或使用 --create_samples 创建示例数据")
        return
    
    if args.validate:
        # 验证数据
        validate_data(args.output_dir)
    
    if args.split:
        # 划分数据集
        print("划分数据集...")
        split_dataset(args.output_dir)
    
    print("数据预处理完成！")

if __name__ == '__main__':
    main() 