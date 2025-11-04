'''
Adversarial Hierarchical-Aware Edge Attention Learning Method for Network Intrusion Detection
@mrforesthao
'''
import torch
import torch.nn.functional as F
from model.SAGE import SAGE, SAGELayer

#新增对抗样本生成模块
class PGDAdversary:
    def __init__(self, model, epsilon=0.1, alpha=0.01, iterations=10):
        """
        :param model: 目标模型
        :param epsilon: 扰动最大幅度
        :param alpha: 单步更新步长
        :param iterations: 攻击迭代次数
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations

    def generate(self, G, node_feat_name='h', edge_feat_name='h', target='coarse', epoch=None):
        """
        生成对抗样本（扰动节点特征）
        :param target: 攻击目标 'coarse'粗粒度,'fine'细粒度,alternate交替攻击
        """
        # 确定当前攻击目标
        if target == 'alternate' and epoch is not None:
            current_target = 'coarse' if epoch % 2 == 0 else 'fine'
        else:
            current_target = target

        # 备份原始特征
        original_nodes = G.ndata[node_feat_name].detach().clone()
        original_edges = G.edata[edge_feat_name].detach().clone()
        
        # 初始化扰动
        delta_nodes = torch.zeros_like(original_nodes, requires_grad=True)
        delta_edges = torch.zeros_like(original_edges, requires_grad=True)
        
        for _ in range(self.iterations):
            # 添加扰动
            G.ndata[node_feat_name] = original_nodes + delta_nodes
            G.edata[edge_feat_name] = original_edges + delta_edges
            
            # 前向计算
            coarse_pred, fine_pred = self.model(G, G.ndata[node_feat_name], G.edata[edge_feat_name])
            
            # 根据目标选择损失
            if current_target == 'coarse':
                loss = F.cross_entropy(coarse_pred, G.edata['Label'])
            elif current_target == 'fine':
                mask = (G.edata['Label'] == 1)
                if mask.sum() == 0:  # 如果没有异常样本
                    loss = torch.tensor(0.0, device=G.device)
                else:
                    loss = F.cross_entropy(fine_pred[mask], G.edata['Attack'][mask])
            else:
                raise ValueError(f"Unknown target: {target}")
            
            # 反向传播获取梯度
            loss.backward()
            
            # 更新扰动
            delta_nodes.data = delta_nodes.data + self.alpha * delta_nodes.grad.sign()
            delta_edges.data = delta_edges.data + self.alpha * delta_edges.grad.sign()
            
            # 投影到ε-ball范围内
            delta_nodes.data = torch.clamp(delta_nodes.data, -self.epsilon, self.epsilon)
            delta_edges.data = torch.clamp(delta_edges.data, -self.epsilon, self.epsilon)
            
            # 梯度清零
            delta_nodes.grad.zero_()
            delta_edges.grad.zero_()
        
        # 恢复原始特征
        G.ndata[node_feat_name] = original_nodes
        G.edata[edge_feat_name] = original_edges
        
        # 返回对抗样本
        adv_G = G.clone()
        adv_G.ndata[node_feat_name] = original_nodes + delta_nodes.detach()
        adv_G.edata[edge_feat_name] = original_edges + delta_edges.detach()
        return adv_G

    
class SafePGDAdversary(PGDAdversary):
    def generate(self, G, target='coarse', epoch=None):
        adv_G = super().generate(G, target, epoch)
        
        # 特征值边界保护（关键！）
        adv_G.ndata['h'] = torch.clamp(adv_G.ndata['h'], 
                                      min=0,  # 假设特征非负
                                      max=10) # 根据数据统计设定
        
        # 边特征保护（如有需要）
        adv_G.edata['h'] = torch.clamp(adv_G.edata['h'], 
                                      min=-1, 
                                      max=1)  # 假设边特征在[-1,1]范围
        return adv_G

#baseline的对抗
class PGDAdversary2:
    def __init__(self, model, epsilon=0.1, alpha=0.01, iterations=10):
        """
        :param model: 目标模型
        :param epsilon: 扰动最大幅度
        :param alpha: 单步更新步长
        :param iterations: 攻击迭代次数
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations

    def generate(self, G, node_feat_name='h', edge_feat_name='h', target='coarse', epoch=None):
        """
        生成对抗样本（扰动节点特征）
        :param target: 攻击目标 'coarse'粗粒度,'fine'细粒度,alternate交替攻击
        """
        # 确定当前攻击目标
        if target == 'alternate' and epoch is not None:
            current_target = 'coarse' if epoch % 2 == 0 else 'fine'
        else:
            current_target = target

        # 备份原始特征
        original_nodes = G.ndata[node_feat_name].detach().clone()
        original_edges = G.edata[edge_feat_name].detach().clone()
        
        # 初始化扰动
        delta_nodes = torch.zeros_like(original_nodes, requires_grad=True)
        delta_edges = torch.zeros_like(original_edges, requires_grad=True)
        
        for _ in range(self.iterations):
            # 添加扰动
            G.ndata[node_feat_name] = original_nodes + delta_nodes
            G.edata[edge_feat_name] = original_edges + delta_edges
            
            # 前向计算
            # coarse_pred, fine_pred = self.model(G, G.ndata[node_feat_name], G.edata[edge_feat_name])
            pred = self.model(G, G.ndata[node_feat_name], G.edata[edge_feat_name])
            # 根据目标选择损失
            if current_target == 'coarse':
                loss = F.cross_entropy(pred, G.edata['Label'])
            elif current_target == 'fine':
                mask = (G.edata['Label'] == 1)
                if mask.sum() == 0:  # 如果没有异常样本
                    loss = torch.tensor(0.0, device=G.device)
                else:
                    loss = F.cross_entropy(pred[mask], G.edata['Attack'][mask])
            else:
                raise ValueError(f"Unknown target: {target}")
            
            # 反向传播获取梯度
            loss.backward()
            
            # 更新扰动
            delta_nodes.data = delta_nodes.data + self.alpha * delta_nodes.grad.sign()
            delta_edges.data = delta_edges.data + self.alpha * delta_edges.grad.sign()
            
            # 投影到ε-ball范围内
            delta_nodes.data = torch.clamp(delta_nodes.data, -self.epsilon, self.epsilon)
            delta_edges.data = torch.clamp(delta_edges.data, -self.epsilon, self.epsilon)
            
            # 梯度清零
            delta_nodes.grad.zero_()
            delta_edges.grad.zero_()
        
        # 恢复原始特征
        G.ndata[node_feat_name] = original_nodes
        G.edata[edge_feat_name] = original_edges
        
        # 返回对抗样本
        adv_G = G.clone()
        adv_G.ndata[node_feat_name] = original_nodes + delta_nodes.detach()
        adv_G.edata[edge_feat_name] = original_edges + delta_edges.detach()
        return adv_G