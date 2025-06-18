import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

class MoNuSegDataset(Dataset):
    """MoNuSeg细胞核分割数据集"""
    
    def __init__(self, data_dir, split='train', transform=None, image_size=512):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # 数据集路径
        self.images_dir = os.path.join(data_dir, 'images')
        self.masks_dir = os.path.join(data_dir, 'masks')
        
        # 获取文件列表
        self.image_files = []
        self.mask_files = []
        
        if os.path.exists(self.images_dir):
            for img_file in os.listdir(self.images_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    img_path = os.path.join(self.images_dir, img_file)
                    mask_path = os.path.join(self.masks_dir, img_file.replace('.jpg', '.png').replace('.jpeg', '.png').replace('.tif', '.png').replace('.tiff', '.png'))
                    
                    if os.path.exists(mask_path):
                        self.image_files.append(img_path)
                        self.mask_files.append(mask_path)
        
        # 如果没有找到数据，创建示例数据
        if len(self.image_files) == 0:
            print(f"警告：在 {data_dir} 中没有找到数据文件，将创建示例数据")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """创建示例数据用于演示"""
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        
        # 创建10个示例图像和掩码
        for i in range(10):
            # 创建随机图像
            img = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
            
            # 创建随机掩码（模拟细胞核）
            mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            
            # 添加一些圆形区域作为细胞核
            for _ in range(np.random.randint(5, 15)):
                center_x = np.random.randint(50, self.image_size - 50)
                center_y = np.random.randint(50, self.image_size - 50)
                radius = np.random.randint(10, 30)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            # 保存图像和掩码
            img_path = os.path.join(self.images_dir, f'sample_{i:03d}.png')
            mask_path = os.path.join(self.masks_dir, f'sample_{i:03d}.png')
            
            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, mask)
            
            self.image_files.append(img_path)
            self.mask_files.append(mask_path)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 调整尺寸
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        
        # 标准化掩码
        mask = (mask > 127).astype(np.uint8) * 255
        
        # 应用变换
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # 基本预处理
            image = image.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / 255.0
            
            # 转换为tensor
            image = torch.from_numpy(image.transpose(2, 0, 1))
            mask = torch.from_numpy(mask).unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'image_path': image_path,
            'mask_path': mask_path
        }

def get_transforms(image_size=512, split='train'):
    """获取数据增强变换"""
    
    if split == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    elif split == 'val':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    else:  # test
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def create_data_loaders(data_dir, batch_size=4, image_size=512, num_workers=4):
    """创建数据加载器"""
    
    # 创建数据集
    train_dataset = MoNuSegDataset(
        data_dir=os.path.join(data_dir, 'train'),
        split='train',
        transform=get_transforms(image_size, 'train'),
        image_size=image_size
    )
    
    val_dataset = MoNuSegDataset(
        data_dir=os.path.join(data_dir, 'val'),
        split='val',
        transform=get_transforms(image_size, 'val'),
        image_size=image_size
    )
    
    test_dataset = MoNuSegDataset(
        data_dir=os.path.join(data_dir, 'test'),
        split='test',
        transform=get_transforms(image_size, 'test'),
        image_size=image_size
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """划分数据集"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 获取所有图像文件
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    
    image_files = []
    for img_file in os.listdir(images_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            image_files.append(img_file)
    
    # 随机打乱
    np.random.shuffle(image_files)
    
    # 计算分割点
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 划分数据集
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # 创建目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(data_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, split, 'masks'), exist_ok=True)
    
    # 移动文件
    def move_files(file_list, split_name):
        for img_file in file_list:
            # 源路径
            src_img = os.path.join(images_dir, img_file)
            mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png').replace('.tif', '.png').replace('.tiff', '.png')
            src_mask = os.path.join(masks_dir, mask_file)
            
            # 目标路径
            dst_img = os.path.join(data_dir, split_name, 'images', img_file)
            dst_mask = os.path.join(data_dir, split_name, 'masks', mask_file)
            
            # 移动文件
            if os.path.exists(src_img):
                os.rename(src_img, dst_img)
            if os.path.exists(src_mask):
                os.rename(src_mask, dst_mask)
    
    move_files(train_files, 'train')
    move_files(val_files, 'val')
    move_files(test_files, 'test')
    
    print(f"数据集划分完成：训练集 {len(train_files)}，验证集 {len(val_files)}，测试集 {len(test_files)}")
    
    # 保存划分信息
    split_info = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    with open(os.path.join(data_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2) 