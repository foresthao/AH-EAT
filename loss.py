'''
Adversarial Hierarchical-Aware Edge Attention Learning Method for Network Intrusion Detection
@mrforesthao
'''
import torch.nn as nn
#自适应层级损失函数
class AdaptiveHierarchicalLoss(nn.Module):
    def __init__(self, temperature=0.5, coarse_class_weights=None, fine_class_weights=None):
        super().__init__()
        self.coarse_criterion = nn.CrossEntropyLoss(weight=coarse_class_weights)  # 假设异常样本占20%
        self.fine_criterion = nn.CrossEntropyLoss(weight=fine_class_weights)
        self.temperature = temperature  # 用于动态调整损失权重

    def forward(self, coarse_pred, fine_pred, coarse_true, fine_true):
        # 粗粒度损失
        coarse_loss = self.coarse_criterion(coarse_pred, coarse_true)
        
        # 细粒度动态加权
        mask = (coarse_true == 1)
        if mask.sum() == 0:
            fine_loss = 0.0
        else:
            # 动态权重计算（基于分类难度）
            with torch.no_grad():
                prob = F.softmax(fine_pred[mask], dim=1)
                entropy = -torch.sum(prob * torch.log(prob), dim=1)
                weights = 1.0 / (1.0 + torch.exp(-self.temperature * entropy))
                weights = weights / weights.sum()
            
            fine_loss = torch.sum(weights * self.fine_criterion(
                fine_pred[mask], fine_true[mask]
            ))
        
        # 自动平衡权重
        total_loss = coarse_loss + (mask.float().mean() * fine_loss)
        loss_dict = {'coarse': coarse_loss.item(), 'fine': fine_loss.item()}
        
        return total_loss, loss_dict

'''分阶段调整权重，前期侧重粗粒度，后期加强细粒度'''
class AdaptiveHierarchicalLoss2(nn.Module):
    def __init__(self, max_epoch=100, coarse_class_weights=None, fine_class_weights=None):
        super().__init__()
        self.coarse_criterion = nn.CrossEntropyLoss(weight=coarse_class_weights)
        self.fine_criterion = nn.CrossEntropyLoss(weight=fine_class_weights)
        self.max_epoch = max_epoch
        self.epoch = 0  # 需在训练循环中更新
        
    def forward(self, coarse_pred, fine_pred, coarse_true, fine_true):
        coarse_loss = self.coarse_criterion(coarse_pred, coarse_true)
        mask = (coarse_true == 1)
        
        if mask.sum() == 0:
            fine_loss = 0.0
        else:
            fine_loss = self.fine_criterion(fine_pred[mask], fine_true[mask])
        
        # 动态调整权重：前期侧重粗粒度，后期平衡
        ratio = self.epoch / self.max_epoch  # 从0到1线性增长
        total_loss = (1 - ratio) * coarse_loss + ratio * fine_loss
        loss_dict = {'coarse': coarse_loss.item(), 'fine': fine_loss.item()}
        
        return total_loss, loss_dict
    
class AdaptiveHierarchicalLoss3(nn.Module):
    '''主要关注细粒度, 与AdaptiveHierarchicalLoss反过来了'''
    def __init__(self, max_epoch=100, coarse_class_weights=None, fine_class_weights=None):
        super().__init__()
        self.coarse_criterion = nn.CrossEntropyLoss(weight=coarse_class_weights)
        self.fine_criterion = nn.CrossEntropyLoss(weight=fine_class_weights)
        self.max_epoch = max_epoch
        self.epoch = 0  # 需在训练循环中更新
        
    def forward(self, coarse_pred, fine_pred, coarse_true, fine_true):
        coarse_loss = self.coarse_criterion(coarse_pred, coarse_true)
        mask = (coarse_true == 1)
        
        if mask.sum() == 0:
            fine_loss = 0.0
        else:
            fine_loss = self.fine_criterion(fine_pred[mask], fine_true[mask])
        
        # 动态调整权重：前期侧重粗粒度，后期平衡
        ratio = self.epoch / self.max_epoch
        total_loss = (1 - ratio) * fine_loss + ratio * coarse_loss
        loss_dict = {'coarse': coarse_loss.item(), 'fine': fine_loss.item()}
        
        return total_loss, loss_dict

class AdaptiveHierarchicalLoss4(nn.Module):
    '''主要关注细粒度90%,其他是粗粒度'''
    def __init__(self, max_epoch=100, coarse_class_weights=None, fine_class_weights=None):
        super().__init__()
        self.coarse_criterion = nn.CrossEntropyLoss(weight=coarse_class_weights)
        self.fine_criterion = nn.CrossEntropyLoss(weight=fine_class_weights)
        self.max_epoch = max_epoch
        self.epoch = 0  # 需在训练循环中更新
        
    def forward(self, coarse_pred, fine_pred, coarse_true, fine_true):
        coarse_loss = self.coarse_criterion(coarse_pred, coarse_true)
        mask = (coarse_true == 1)
        
        if mask.sum() == 0:
            fine_loss = 0.0
        else:
            fine_loss = self.fine_criterion(fine_pred[mask], fine_true[mask])
        
        # 主要考虑细粒度
        ratio_fine = 0.9

        total_loss = (1 - ratio_fine) * coarse_loss + ratio_fine * fine_loss
        loss_dict = {'coarse': coarse_loss.item(), 'fine': fine_loss.item()}
        
        return total_loss, loss_dict

class AdaptiveHierarchicalLoss5(nn.Module):
    '''主要关注细粒度100%,其他是粗粒度'''
    def __init__(self, max_epoch=100, coarse_class_weights=None, fine_class_weights=None):
        super().__init__()
        self.coarse_criterion = nn.CrossEntropyLoss(weight=coarse_class_weights)
        self.fine_criterion = nn.CrossEntropyLoss(weight=fine_class_weights)
        self.max_epoch = max_epoch
        self.epoch = 0  # 需在训练循环中更新
        
    def forward(self, coarse_pred, fine_pred, coarse_true, fine_true):
        # coarse_loss = self.coarse_criterion(coarse_pred, coarse_true)
        coarse_loss = 0.0
        mask = (coarse_true == 1)
        
        if mask.sum() == 0:
            fine_loss = 0.0
        else:
            fine_loss = self.fine_criterion(fine_pred[mask], fine_true[mask])
        
        # 主要考虑细粒度
        ratio_fine = 1

        total_loss = (1 - ratio_fine) * coarse_loss + ratio_fine * fine_loss
        # loss_dict = {'coarse': coarse_loss.item(), 'fine': fine_loss.item()}
        loss_dict = {'coarse': 0.0, 'fine': fine_loss.item()}
        
        return total_loss, loss_dict