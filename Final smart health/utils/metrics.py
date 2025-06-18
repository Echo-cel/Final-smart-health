import torch
import numpy as np
from sklearn.metrics import jaccard_score
import cv2

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    计算Dice系数
    
    Args:
        y_true: 真实标签 (B, 1, H, W) 或 (B, H, W)
        y_pred: 预测结果 (B, 1, H, W) 或 (B, H, W)
        smooth: 平滑因子，避免除零
    
    Returns:
        dice: Dice系数
    """
    # 确保输入是tensor
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    
    # 确保维度正确
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    if y_pred.dim() == 3:
        y_pred = y_pred.unsqueeze(1)
    
    # 二值化预测结果
    y_pred = torch.sigmoid(y_pred) if y_pred.max() > 1 else y_pred
    y_pred = (y_pred > 0.5).float()
    
    # 展平
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    # 计算交集
    intersection = (y_true_flat * y_pred_flat).sum()
    
    # 计算并集
    union = y_true_flat.sum() + y_pred_flat.sum()
    
    # 计算Dice系数
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.item()

def iou_score(y_true, y_pred, smooth=1e-6):
    """
    计算IoU (Intersection over Union)
    
    Args:
        y_true: 真实标签 (B, 1, H, W) 或 (B, H, W)
        y_pred: 预测结果 (B, 1, H, W) 或 (B, H, W)
        smooth: 平滑因子，避免除零
    
    Returns:
        iou: IoU分数
    """
    # 确保输入是tensor
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    
    # 确保维度正确
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    if y_pred.dim() == 3:
        y_pred = y_pred.unsqueeze(1)
    
    # 二值化预测结果
    y_pred = torch.sigmoid(y_pred) if y_pred.max() > 1 else y_pred
    y_pred = (y_pred > 0.5).float()
    
    # 展平
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    # 计算交集
    intersection = (y_true_flat * y_pred_flat).sum()
    
    # 计算并集
    union = y_true_flat.sum() + y_pred_flat.sum() - intersection
    
    # 计算IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()

def precision_score(y_true, y_pred, smooth=1e-6):
    """
    计算精确率
    
    Args:
        y_true: 真实标签
        y_pred: 预测结果
        smooth: 平滑因子
    
    Returns:
        precision: 精确率
    """
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    if y_pred.dim() == 3:
        y_pred = y_pred.unsqueeze(1)
    
    y_pred = torch.sigmoid(y_pred) if y_pred.max() > 1 else y_pred
    y_pred = (y_pred > 0.5).float()
    
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    intersection = (y_true_flat * y_pred_flat).sum()
    predicted_positive = y_pred_flat.sum()
    
    precision = (intersection + smooth) / (predicted_positive + smooth)
    
    return precision.item()

def recall_score(y_true, y_pred, smooth=1e-6):
    """
    计算召回率
    
    Args:
        y_true: 真实标签
        y_pred: 预测结果
        smooth: 平滑因子
    
    Returns:
        recall: 召回率
    """
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    
    if y_true.dim() == 3:
        y_true = y_true.unsqueeze(1)
    if y_pred.dim() == 3:
        y_pred = y_pred.unsqueeze(1)
    
    y_pred = torch.sigmoid(y_pred) if y_pred.max() > 1 else y_pred
    y_pred = (y_pred > 0.5).float()
    
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    intersection = (y_true_flat * y_pred_flat).sum()
    actual_positive = y_true_flat.sum()
    
    recall = (intersection + smooth) / (actual_positive + smooth)
    
    return recall.item()

def f1_score(y_true, y_pred, smooth=1e-6):
    """
    计算F1分数
    
    Args:
        y_true: 真实标签
        y_pred: 预测结果
        smooth: 平滑因子
    
    Returns:
        f1: F1分数
    """
    precision = precision_score(y_true, y_pred, smooth)
    recall = recall_score(y_true, y_pred, smooth)
    
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    
    return f1

def calculate_metrics(y_true, y_pred):
    """
    计算所有评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测结果
    
    Returns:
        metrics: 包含所有指标的字典
    """
    dice = dice_coefficient(y_true, y_pred)
    iou = iou_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics = {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def calculate_metrics_batch(y_true_batch, y_pred_batch):
    """
    计算批次数据的平均指标
    
    Args:
        y_true_batch: 批次真实标签 (B, 1, H, W)
        y_pred_batch: 批次预测结果 (B, 1, H, W)
    
    Returns:
        avg_metrics: 平均指标
    """
    batch_size = y_true_batch.size(0)
    total_metrics = {
        'dice': 0.0,
        'iou': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    }
    
    for i in range(batch_size):
        y_true = y_true_batch[i:i+1]
        y_pred = y_pred_batch[i:i+1]
        
        metrics = calculate_metrics(y_true, y_pred)
        
        for key in total_metrics:
            total_metrics[key] += metrics[key]
    
    # 计算平均值
    avg_metrics = {key: value / batch_size for key, value in total_metrics.items()}
    
    return avg_metrics

def post_process_prediction(pred, threshold=0.5):
    """
    后处理预测结果
    
    Args:
        pred: 模型预测输出
        threshold: 二值化阈值
    
    Returns:
        processed_pred: 处理后的预测结果
    """
    # 应用sigmoid激活函数
    if pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    # 二值化
    pred = (pred > threshold).float()
    
    return pred

def visualize_prediction(image, true_mask, pred_mask, save_path=None):
    """
    可视化预测结果
    
    Args:
        image: 原始图像 (H, W, 3)
        true_mask: 真实掩码 (H, W)
        pred_mask: 预测掩码 (H, W)
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 真实掩码
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('真实掩码')
    axes[1].axis('off')
    
    # 预测掩码
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('预测掩码')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 