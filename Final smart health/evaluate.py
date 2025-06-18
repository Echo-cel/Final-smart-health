import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 导入自定义模块
from models.transunet import TransUNet
from utils.dataset import create_data_loaders
from utils.metrics import calculate_metrics_batch, post_process_prediction, visualize_prediction

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_model(checkpoint_path, config, device):
    """加载训练好的模型"""
    # 创建模型
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
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"加载模型检查点: {checkpoint_path}")
    print(f"训练轮次: {checkpoint['epoch']}")
    print(f"验证损失: {checkpoint['loss']:.4f}")
    print(f"验证指标: {checkpoint['metrics']}")
    
    return model

def evaluate_model(model, test_loader, device, config):
    """评估模型性能"""
    model.eval()
    
    all_metrics = []
    all_predictions = []
    all_targets = []
    
    print("开始评估模型...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="评估进度")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 后处理预测结果
            predictions = post_process_prediction(outputs)
            
            # 计算指标
            metrics = calculate_metrics_batch(masks, predictions)
            all_metrics.append(metrics)
            
            # 保存预测结果和目标
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(masks.cpu().numpy())
    
    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics, all_predictions, all_targets

def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制混淆矩阵"""
    # 展平数据
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['背景', '细胞核'], 
                yticklabels=['背景', '细胞核'])
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_comparison(metrics, save_path):
    """绘制指标对比图"""
    metrics_names = list(metrics.keys())
    metrics_values = list(metrics.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.title('模型性能指标对比')
    plt.ylabel('分数')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_sample_results(model, test_loader, device, config, num_samples=5):
    """可视化样本结果"""
    model.eval()
    
    # 获取样本数据
    sample_batch = next(iter(test_loader))
    images = sample_batch['image'][:num_samples].to(device)
    masks = sample_batch['mask'][:num_samples].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        predictions = post_process_prediction(outputs)
    
    # 可视化结果
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # 原始图像
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'样本 {i+1} - 原始图像')
        axes[i, 0].axis('off')
        
        # 真实掩码
        true_mask = masks[i, 0].cpu().numpy()
        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title(f'样本 {i+1} - 真实掩码')
        axes[i, 1].axis('off')
        
        # 预测掩码
        pred_mask = predictions[i, 0].cpu().numpy()
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title(f'样本 {i+1} - 预测掩码')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(config['paths']['results_dir'], 'sample_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"样本结果已保存到: {save_path}")

def save_evaluation_results(metrics, config):
    """保存评估结果"""
    # 创建结果DataFrame
    results_df = pd.DataFrame([metrics])
    
    # 保存到CSV
    csv_path = os.path.join(config['paths']['results_dir'], 'evaluation_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    # 保存详细结果到文本文件
    txt_path = os.path.join(config['paths']['results_dir'], 'evaluation_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("模型评估结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"数据集: {config['dataset']['name']}\n")
        f.write(f"模型: {config['model']['name']}\n")
        f.write(f"图像尺寸: {config['dataset']['image_size']}x{config['dataset']['image_size']}\n")
        f.write("\n性能指标:\n")
        f.write("-" * 30 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
    
    print(f"评估结果已保存到: {csv_path}")
    print(f"详细结果已保存到: {txt_path}")

def main():
    """主评估函数"""
    # 加载配置
    config = load_config('configs/config.yaml')
    
    # 设置设备
    device = torch.device(config['misc']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("创建测试数据加载器...")
    _, _, test_loader = create_data_loaders(
        data_dir=config['paths']['data_dir'],
        batch_size=1,  # 评估时使用batch_size=1
        image_size=config['dataset']['image_size'],
        num_workers=config['misc']['num_workers']
    )
    
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 加载模型
    checkpoint_path = os.path.join(config['paths']['checkpoints_dir'], 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(config['paths']['checkpoints_dir'], 'latest_checkpoint.pth')
    
    if not os.path.exists(checkpoint_path):
        print("错误：找不到模型检查点文件！")
        print("请先运行训练脚本：python train.py")
        return
    
    model = load_model(checkpoint_path, config, device)
    
    # 评估模型
    metrics, predictions, targets = evaluate_model(model, test_loader, device, config)
    
    # 打印评估结果
    print("\n" + "="*50)
    print("模型评估结果")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    print("="*50)
    
    # 保存评估结果
    save_evaluation_results(metrics, config)
    
    # 绘制混淆矩阵
    cm_path = os.path.join(config['paths']['results_dir'], 'confusion_matrix.png')
    plot_confusion_matrix(targets, predictions, cm_path)
    
    # 绘制指标对比图
    metrics_path = os.path.join(config['paths']['results_dir'], 'metrics_comparison.png')
    plot_metrics_comparison(metrics, metrics_path)
    
    # 可视化样本结果
    visualize_sample_results(model, test_loader, device, config)
    
    print("\n评估完成！")
    print(f"结果文件保存在: {config['paths']['results_dir']}")

if __name__ == '__main__':
    main() 