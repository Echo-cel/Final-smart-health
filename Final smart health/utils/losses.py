import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice损失函数"""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        # 确保输入是浮点型
        y_pred = y_pred.float()
        y_true = y_true.float()
        
        # 应用sigmoid激活函数
        y_pred = torch.sigmoid(y_pred)
        
        # 展平
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        # 计算交集
        intersection = (y_pred_flat * y_true_flat).sum()
        
        # 计算并集
        union = y_pred_flat.sum() + y_true_flat.sum()
        
        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 返回损失（1 - Dice系数）
        return 1 - dice

class BCELoss(nn.Module):
    """二元交叉熵损失函数"""
    
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, y_pred, y_true):
        return self.bce(y_pred, y_true)

class DiceBCELoss(nn.Module):
    """Dice + BCE组合损失函数"""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, y_pred, y_true):
        # BCE损失
        bce_loss = self.bce(y_pred, y_true)
        
        # Dice损失
        y_pred_sigmoid = torch.sigmoid(y_pred)
        y_pred_flat = y_pred_sigmoid.view(-1)
        y_true_flat = y_true.view(-1)
        
        intersection = (y_pred_flat * y_true_flat).sum()
        union = y_pred_flat.sum() + y_true_flat.sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 组合损失
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss

class FocalLoss(nn.Module):
    """Focal损失函数，用于处理类别不平衡"""
    
    def __init__(self, alpha=1, gamma=2, smooth=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        # 应用sigmoid
        y_pred_sigmoid = torch.sigmoid(y_pred)
        
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        
        # 计算pt
        pt = y_pred_sigmoid * y_true + (1 - y_pred_sigmoid) * (1 - y_true)
        
        # 计算focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

class IoULoss(nn.Module):
    """IoU损失函数"""
    
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        # 确保输入是浮点型
        y_pred = y_pred.float()
        y_true = y_true.float()
        
        # 应用sigmoid激活函数
        y_pred = torch.sigmoid(y_pred)
        
        # 展平
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        # 计算交集
        intersection = (y_pred_flat * y_true_flat).sum()
        
        # 计算并集
        union = y_pred_flat.sum() + y_true_flat.sum() - intersection
        
        # 计算IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # 返回损失（1 - IoU）
        return 1 - iou

class ComboLoss(nn.Module):
    """组合损失函数：Dice + BCE + IoU"""
    
    def __init__(self, dice_weight=0.3, bce_weight=0.3, iou_weight=0.4, smooth=1e-6):
        super(ComboLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, y_pred, y_true):
        # BCE损失
        bce_loss = self.bce(y_pred, y_true)
        
        # Dice损失
        y_pred_sigmoid = torch.sigmoid(y_pred)
        y_pred_flat = y_pred_sigmoid.view(-1)
        y_true_flat = y_true.view(-1)
        
        intersection = (y_pred_flat * y_true_flat).sum()
        union = y_pred_flat.sum() + y_true_flat.sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        
        # IoU损失
        union_iou = y_pred_flat.sum() + y_true_flat.sum() - intersection
        iou_loss = 1 - (intersection + self.smooth) / (union_iou + self.smooth)
        
        # 组合损失
        total_loss = (self.bce_weight * bce_loss + 
                     self.dice_weight * dice_loss + 
                     self.iou_weight * iou_loss)
        
        return total_loss

def get_loss_function(loss_name='dice_bce'):
    """
    获取损失函数
    
    Args:
        loss_name: 损失函数名称
    
    Returns:
        loss_function: 损失函数
    """
    if loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'bce':
        return BCELoss()
    elif loss_name == 'dice_bce':
        return DiceBCELoss()
    elif loss_name == 'focal':
        return FocalLoss()
    elif loss_name == 'iou':
        return IoULoss()
    elif loss_name == 'combo':
        return ComboLoss()
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}") 