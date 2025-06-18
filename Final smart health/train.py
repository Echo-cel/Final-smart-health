import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from datetime import datetime

# 导入自定义模块
from models.transunet import TransUNet
from utils.dataset import create_data_loaders, split_dataset
from utils.metrics import calculate_metrics_batch
from utils.losses import get_loss_function

def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def create_directories(config):
    """创建必要的目录"""
    directories = [
        config['paths']['checkpoints_dir'],
        config['paths']['logs_dir'],
        config['paths']['results_dir'],
        config['paths']['data_dir'],
        config['paths']['raw_data_dir'],
        config['paths']['processed_data_dir'],
        config['paths']['splits_dir']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} - Training')
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算指标
        with torch.no_grad():
            metrics = calculate_metrics_batch(masks, outputs)
        
        # 更新统计
        total_loss += loss.item()
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Dice': f'{metrics["dice"]:.4f}',
            'IoU': f'{metrics["iou"]:.4f}'
        })
    
    # 计算平均值
    avg_loss = total_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics

def validate_epoch(model, val_loader, criterion, device, epoch):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    total_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    num_batches = len(val_loader)
    
    progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1} - Validation')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 计算指标
            metrics = calculate_metrics_batch(masks, outputs)
            
            # 更新统计
            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{metrics["dice"]:.4f}',
                'IoU': f'{metrics["iou"]:.4f}'
            })
    
    # 计算平均值
    avg_loss = total_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics

def save_checkpoint(model, optimizer, epoch, loss, metrics, config, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': config
    }
    
    # 保存最新检查点
    checkpoint_path = os.path.join(config['paths']['checkpoints_dir'], 'latest_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 如果是最佳模型，保存最佳检查点
    if is_best:
        best_checkpoint_path = os.path.join(config['paths']['checkpoints_dir'], 'best_checkpoint.pth')
        torch.save(checkpoint, best_checkpoint_path)
        print(f"保存最佳模型，Dice: {metrics['dice']:.4f}")

def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, config):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(epochs, train_losses, 'b-', label='训练损失')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='验证损失')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice系数曲线
    axes[0, 1].plot(epochs, [m['dice'] for m in train_metrics], 'b-', label='训练Dice')
    axes[0, 1].plot(epochs, [m['dice'] for m in val_metrics], 'r-', label='验证Dice')
    axes[0, 1].set_title('Dice系数曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice系数')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU曲线
    axes[1, 0].plot(epochs, [m['iou'] for m in train_metrics], 'b-', label='训练IoU')
    axes[1, 0].plot(epochs, [m['iou'] for m in val_metrics], 'r-', label='验证IoU')
    axes[1, 0].set_title('IoU曲线')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1分数曲线
    axes[1, 1].plot(epochs, [m['f1'] for m in train_metrics], 'b-', label='训练F1')
    axes[1, 1].plot(epochs, [m['f1'] for m in val_metrics], 'r-', label='验证F1')
    axes[1, 1].set_title('F1分数曲线')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1分数')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    plot_path = os.path.join(config['paths']['results_dir'], 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"训练曲线已保存到: {plot_path}")

def main():
    """主训练函数"""
    # 加载配置
    config = load_config('configs/config.yaml')
    
    # 设置随机种子
    set_seed(config['misc']['seed'])
    
    # 创建目录
    create_directories(config)
    
    # 设置设备
    device = torch.device(config['misc']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config['paths']['data_dir'],
        batch_size=config['training']['batch_size'],
        image_size=config['dataset']['image_size'],
        num_workers=config['misc']['num_workers']
    )
    
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    print("创建TransUNet模型...")
    model = TransUNet(
        img_size=config['dataset']['image_size'],
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        embed_dim=config['model']['embed_dim'],
        patch_size=config['model']['patch_size'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        mlp_ratio=config['model']['mlp_ratio'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建损失函数
    criterion = get_loss_function(config['training']['loss_function'])
    
    # 创建优化器
    if config['training']['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    
    # 创建学习率调度器
    if config['training']['scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    
    # 创建TensorBoard写入器
    log_dir = os.path.join(config['paths']['logs_dir'], datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    print("开始训练...")
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    best_dice = 0.0
    
    for epoch in range(config['training']['num_epochs']):
        # 训练
        train_loss, train_epoch_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_epoch_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Dice/Train', train_epoch_metrics['dice'], epoch)
        writer.add_scalar('Dice/Val', val_epoch_metrics['dice'], epoch)
        writer.add_scalar('IoU/Train', train_epoch_metrics['iou'], epoch)
        writer.add_scalar('IoU/Val', val_epoch_metrics['iou'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存历史记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics.append(train_epoch_metrics)
        val_metrics.append(val_epoch_metrics)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"  训练 - 损失: {train_loss:.4f}, Dice: {train_epoch_metrics['dice']:.4f}, IoU: {train_epoch_metrics['iou']:.4f}")
        print(f"  验证 - 损失: {val_loss:.4f}, Dice: {val_epoch_metrics['dice']:.4f}, IoU: {val_epoch_metrics['iou']:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存检查点
        is_best = val_epoch_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_epoch_metrics['dice']
        
        if (epoch + 1) % config['misc']['save_freq'] == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, val_loss, val_epoch_metrics, config, is_best)
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, config)
    
    print("训练完成！")
    print(f"最佳验证Dice系数: {best_dice:.4f}")

if __name__ == '__main__':
    main() 