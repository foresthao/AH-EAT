'''
Adversarial Hierarchical-Aware Edge Attention Learning Method for Network Intrusion Detection
@mrforesthao
'''
import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from model.Attention import HierarchicalGAT, HierarchicalGAT1
from model.adversarial import PGDAdversary, SafePGDAdversary
from dgl.dataloading import MultiLayerNeighborSampler, EdgeDataLoader

def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()

def compute_f1_score(pred, labels):
    pred_labels = pred.argmax(1).cpu().numpy()
    true_labels = labels.cpu().numpy()
    return f1_score(true_labels, pred_labels, average='weighted')

def load_data(dataset_path, device):
    """ 加载并处理层级化标签,Label是粗粒度标签, Attack是细粒度标签"""
    with open(dataset_path, 'rb') as f:
        G = pickle.load(f)
    
    # 确保存在细粒度标签字段
    assert 'Attack' in G.edata,"需要细粒度攻击类型标签"
    assert 'Label' in G.edata,"需要粗粒度攻击类型标签"

    
    # 维度调整
    G.ndata['h'] = G.ndata['h'].unsqueeze(1)  # [N,1,feat_dim]
    G.edata['h'] = G.edata['h'].unsqueeze(1)  # [E,1,feat_dim]
    
    G = G.to(device)
    return G

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
        coarse_loss = self.coarse_criterion(coarse_pred, coarse_true)
        mask = (coarse_true == 1)
        
        if mask.sum() == 0:
            fine_loss = 0.0
        else:
            fine_loss = self.fine_criterion(fine_pred[mask], fine_true[mask])
        
        # 主要考虑细粒度
        ratio_fine = 1

        total_loss = (1 - ratio_fine) * coarse_loss + ratio_fine * fine_loss
        loss_dict = {'coarse': coarse_loss.item(), 'fine': fine_loss.item()}
        
        return total_loss, loss_dict


def train(model, G, device, args):
    # 0.检查图是否在GPU上
    print(f"Graph device: {G.device}")
    print(f"Features device: {G.ndata['h'].device}")
    
    # 1. 计算细粒度类别权重
    fine_labels = G.edata['Attack'].cpu().numpy()
    fine_weights = class_weight.compute_class_weight(
        'balanced', 
        classes=np.unique(fine_labels), 
        y=fine_labels
    )
    fine_weights = torch.FloatTensor(fine_weights).to(device)
    
    # 2. 初始化优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs
    )
    
    # 3. 初始化损失函数 
    criterion = AdaptiveHierarchicalLoss4(
        max_epoch = args.num_epochs,
        coarse_class_weights=torch.tensor([1.0, 5.0]),
        fine_class_weights=fine_weights
    ).to(device)
    
    # 4. 数据准备
    train_mask = G.edata['train_mask']
    coarse_labels = G.edata['Label']
    fine_labels = G.edata['Attack']
    
    # 5. 初始化对抗生成器（更保守的参数）
    pgd_adversary = PGDAdversary(
        model=model,
        epsilon=0.05,  # 减小扰动幅度
        alpha=0.005,   # 减小步长
        iterations=5   # 减少迭代次数
    )

    # 6. 稳定性监控参数
    best_loss = float('inf')
    patience = 20  # 连续10次loss不下降则停止训练
    bad_epochs = 0


    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()

        # ================== 对抗样本生成 ==================
        adv_G = pgd_adversary.generate(G, target='alternate', epoch=epoch)
        
        # ================== 前向计算 ==================
        # 原始样本
        coarse_pred, fine_pred = model(G, G.ndata['h'], G.edata['h'])
        # 对抗样本
        adv_coarse, adv_fine = model(adv_G, adv_G.ndata['h'], adv_G.edata['h'])
        
        # ================== 损失计算 ==================
        # 合并预测结果
        combined_coarse = torch.cat([coarse_pred, adv_coarse], dim=0)
        combined_fine = torch.cat([fine_pred, adv_fine], dim=0)
        combined_labels_coarse = torch.cat([coarse_labels, coarse_labels], dim=0)
        combined_labels_fine = torch.cat([fine_labels, fine_labels], dim=0)
        combined_mask = torch.cat([train_mask, train_mask], dim=0)
        
        # 数值稳定性保护
        combined_coarse = torch.clamp(combined_coarse, min=-50, max=50)
        combined_fine = torch.clamp(combined_fine, min=-50, max=50)
        
        loss, loss_dict = criterion(
            combined_coarse[combined_mask],
            combined_fine[combined_mask],
            combined_labels_coarse[combined_mask],
            combined_labels_fine[combined_mask]
        )
        
        # ================== 反向传播与优化 ==================
        # 梯度稳定性检查
        if torch.isnan(loss):
            print(f"Epoch {epoch}: 检测到NaN损失，跳过该批次")
            continue
            
        loss.backward()
        
        # 梯度裁剪（关键！）
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=2.0, 
            error_if_nonfinite=False
        )
        
        # 参数更新
        optimizer.step()
        scheduler.step()
        
        # ================== 稳定性监控 ==================
        # 提前停止机制，准备先停止
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            bad_epochs = 0
        # else:
        #     bad_epochs += 1
        #     if bad_epochs >= patience:
        #         print(f"Early stopping at epoch {epoch}")
        #         break
                
        # ================== 训练监控 ==================
        if epoch % 50 == 0:
            # 学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 粗粒度评估
            with torch.no_grad():
                coarse_acc = compute_accuracy(coarse_pred[train_mask], coarse_labels[train_mask])
                coarse_f1 = compute_f1_score(coarse_pred[train_mask], coarse_labels[train_mask])
                
                # 细粒度评估
                # mask = (coarse_labels[train_mask] == 1)
                # fine_acc = compute_accuracy(fine_pred[train_mask][mask], fine_labels[train_mask][mask])
                # fine_f1 = compute_f1_score(fine_pred[train_mask][mask], fine_labels[train_mask][mask])
                fine_acc = compute_accuracy(fine_pred[train_mask], fine_labels[train_mask])
                fine_f1 = compute_f1_score(fine_pred[train_mask], fine_labels[train_mask])

            
            # 打印监控信息
            print(f"Epoch {epoch:03d} | "
                  f"Loss: {current_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Coarse Acc: {coarse_acc:.4f} | Coarse F1: {coarse_f1:.4f} |"
                  f"Fine Acc: {fine_acc:.4f} | Fine F1: {fine_f1:.4f}")

def train1(model, G, device, args):
    #去掉
    # 0.检查图是否在GPU上
    print('不调整lr')
    print(f"Graph device: {G.device}")
    print(f"Features device: {G.ndata['h'].device}")
    
    # 1. 计算细粒度类别权重
    fine_labels = G.edata['Attack'].cpu().numpy()
    fine_weights = class_weight.compute_class_weight(
        'balanced', 
        classes=np.unique(fine_labels), 
        y=fine_labels
    )
    fine_weights = torch.FloatTensor(fine_weights).to(device)
    
    # 2. 初始化优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # 3. 初始化损失函数
    # criterion = AdaptiveHierarchicalLoss(
    #     temperature=0.5, 
    #     coarse_class_weights=torch.tensor([1.0, 5.0]),
    #     fine_class_weights=fine_weights
    # ).to(device)
    
    criterion = AdaptiveHierarchicalLoss5(
        max_epoch = args.num_epochs,
        coarse_class_weights=torch.tensor([1.0, 5.0]),
        fine_class_weights=fine_weights
    ).to(device)
    
    # 4. 数据准备
    train_mask = G.edata['train_mask']
    coarse_labels = G.edata['Label']
    fine_labels = G.edata['Attack']
    
    # 5. 初始化对抗生成器（更保守的参数）
    pgd_adversary = PGDAdversary(
        model=model,
        epsilon=0.05,  # 减小扰动幅度
        alpha=0.005,   # 减小步长
        iterations=5   # 减少迭代次数
    )



    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()

        # ================== 对抗样本生成 ==================
        adv_G = pgd_adversary.generate(G, target='alternate', epoch=epoch)
        
        # ================== 前向计算 ==================
        # 原始样本
        coarse_pred, fine_pred = model(G, G.ndata['h'], G.edata['h'])
        # 对抗样本
        adv_coarse, adv_fine = model(adv_G, adv_G.ndata['h'], adv_G.edata['h'])
        
        # ================== 损失计算 ==================
        # 合并预测结果
        combined_coarse = torch.cat([coarse_pred, adv_coarse], dim=0)
        combined_fine = torch.cat([fine_pred, adv_fine], dim=0)
        combined_labels_coarse = torch.cat([coarse_labels, coarse_labels], dim=0)
        combined_labels_fine = torch.cat([fine_labels, fine_labels], dim=0)
        combined_mask = torch.cat([train_mask, train_mask], dim=0)
        
        
        loss, loss_dict = criterion(
            combined_coarse[combined_mask],
            combined_fine[combined_mask],
            combined_labels_coarse[combined_mask],
            combined_labels_fine[combined_mask]
        )

            
        loss.backward()
        optimizer.step()
        
        # ================== 训练监控 ==================
        if epoch % 50 == 0:
            # 学习率
            current_lr = optimizer.param_groups[0]['lr']
            current_loss = loss.item()
            # 粗粒度评估
            with torch.no_grad():
                # coarse_acc = (coarse_pred[train_mask].argmax(1) == coarse_labels[train_mask]).float().mean()
                coarse_acc = compute_accuracy(coarse_pred[train_mask], coarse_labels[train_mask])
                coarse_f1 = compute_f1_score(coarse_pred[train_mask], coarse_labels[train_mask])
                
                # 细粒度评估
                mask = (coarse_labels[train_mask] == 1)
                if mask.sum() > 0:
                    # fine_acc = (fine_pred[train_mask][mask].argmax(1) == fine_labels[train_mask][mask]).float().mean()
                    fine_acc = compute_accuracy(fine_pred[train_mask][mask], fine_labels[train_mask][mask])
                    fine_f1 = compute_f1_score(fine_pred[train_mask][mask], fine_labels[train_mask][mask])
                else:
                    fine_acc = torch.tensor(0.0)
                    fine_f1 = torch.tensor(0.0)
            
            # 打印监控信息
            print(f"Epoch {epoch:03d} | "
                  f"Loss: {current_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Coarse Acc: {coarse_acc:.4f} | Coarse F1: {coarse_f1:.4f} |"
                  f"Fine Acc: {fine_acc:.4f} | Fine F1: {fine_f1:.4f}")

def evaluate(model, G_test, device):
    '''在测试集上评估模型性能'''
    model.eval()
    with torch.no_grad():
        # 获取测试掩码（测试图的所有边均为测试样本）
        test_mask = G_test.edata.get(
            'test_mask', 
            torch.ones(G_test.num_edges(), dtype=torch.bool, device=device)
        )

        # 向前传播
        coarse_pred, fine_pred = model(G_test, G_test.ndata['h'], G_test.edata['h'])

        # 粗粒度准确里
        coarse_labels = G_test.edata['Label'][test_mask]
        # coarse_acc = (coarse_pred[test_mask].argmax(1) == coarse_labels).float().mean().item()
        coarse_acc = compute_accuracy(coarse_pred[test_mask], coarse_labels)
        coarse_f1 = compute_f1_score(coarse_pred[test_mask], coarse_labels)

        # 细粒度准确率（仅在粗粒度预测为攻击时计算）
        fine_labels = G_test.edata['Attack'][test_mask]
        fine_acc = compute_accuracy(fine_pred[test_mask], fine_labels)
        fine_f1 = compute_f1_score(fine_pred[test_mask], fine_labels)

        return coarse_acc, coarse_f1, fine_acc, fine_f1
    

def evaluate_adv(model, G_test, device, adversary, target='coarse', epoch=None):
    '''在测试集上评估模型性能，使用对抗样本'''
    model.eval()
    # 生成对抗样本
    adv_G = adversary.generate(G_test, target=target, epoch=epoch)

    with torch.no_grad():
        # 获取测试掩码（测试图的所有边均为测试样本）
        test_mask = adv_G.edata.get(
            'test_mask', 
            torch.ones(adv_G.num_edges(), dtype=torch.bool, device=device)
        )

        # 向前传播
        coarse_pred, fine_pred = model(adv_G, adv_G.ndata['h'], adv_G.edata['h'])

        # 粗粒度准确率
        coarse_labels = adv_G.edata['Label'][test_mask]
        coarse_acc = compute_accuracy(coarse_pred[test_mask], coarse_labels)
        coarse_f1 = compute_f1_score(coarse_pred[test_mask], coarse_labels)

        # 细粒度准确率（仅在粗粒度预测为攻击时计算）
        mask = (coarse_labels == 1)
        if mask.sum() > 0:
            fine_labels = adv_G.edata['Attack'][test_mask][mask]
            fine_acc = compute_accuracy(fine_pred[test_mask][mask], fine_labels)
            fine_f1 = compute_f1_score(fine_pred[test_mask][mask], fine_labels)
        else:
            fine_acc = 0.0
            fine_f1 = 0.0

    return coarse_acc, coarse_f1, fine_acc, fine_f1

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data2_path = '/data2/yanhao_data/e_gat/dataset/graph/'
    print('1.加载保存的数据集')
    if args.dataset == 'BoT-IoT':
        G = load_data(os.path.join(data2_path,'./saved_graph_nf_bot_iot.pkl'), device)
        G_test = load_data(os.path.join(data2_path,'./saved_graph_nf_bot_iot_test.pkl'), device)
        num_fine_class = 5
    elif args.dataset == 'ToN-IoT':
        G = load_data(os.path.join(data2_path,'./saved_graph_nf_ton.pkl'), device)
        G_test = load_data(os.path.join(data2_path,'./saved_graph_nf_ton_test.pkl'), device)
        num_fine_class = 10
    elif args.dataset == 'CIC-IDS2018':
        G = load_data(os.path.join(data2_path,'./saved_graph_cic_ids18_10%.pkl'), device)
        G_test = load_data(os.path.join(data2_path,'./saved_graph_cic_ids18_10%_test.pkl'), device)
        num_fine_class = 15
    elif args.dataset == 'UNSW-NB15':
        G = load_data(os.path.join(data2_path,'./saved_graph_nf_nb15.pkl'), device)
        G_test = load_data(os.path.join(data2_path,'./saved_graph_nf_nb15_test.pkl'), device)
        num_fine_class = 10
    else:
        print('error dataset!!!')

    #建立模型
    model = HierarchicalGAT1(node_feat_dim=G.ndata['h'].shape[2],
                            edge_feat_dim=G.edata['h'].shape[2],
                            num_fine_classes=num_fine_class).to(device)
    
    print('2.开始训练')
    # 调用train方法
    train(model=model, G=G, device=device, args=args)
    # train1(model=model, G=G, device=device, args=args)

    adversary = PGDAdversary(
        model=model,
        epsilon=0.05,  # 扰动幅度
        alpha=0.005,   # 步长
        iterations=10   # 迭代次数
    )

    print('3.在测试集上评估最终模型')
    test_coarse_acc, test_coarse_f1, test_fine_acc, test_fine_f1 = evaluate(model, G_test, device)
    # test_coarse_acc, test_coarse_f1, test_fine_acc, test_fine_f1 = evaluate_adv(model, G_test, device, adversary, target='fine')
    print(f"[Final Test] 粗粒度准确率: {test_coarse_acc:.4f}, 粗粒度f1:{test_coarse_f1:.4f}, 细粒度准确率: {test_fine_acc:.4f}, 细粒度f1:{test_fine_f1:.4f}")

if __name__ == '__main__':
    #1.导入数据，分别是具有粗细两种粒度的图
    parser = argparse.ArgumentParser(description='E-GAT_hierarchical model')
    parser.add_argument('--dataset', type=str, default='BoT-IoT', choices=['BoT-IoT', 'ToN-IoT', 'CIC-IDS2018', 'UNSW-NB15'])
    parser.add_argument("--num_runs", type=int, default=1, help="number of runs, default = 10")
    parser.add_argument("--num_epochs", type=int, default=2001, help="number of training epochs, default = 50")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0, help="weight decay")
    parser.add_argument('--gnn', type=str, default='GAT', help="Graph attention network")
    parser.add_argument('--batch_size', type=int, default=1024, help="batch size for training")

    args = parser.parse_args()
    
    print(args)
    run(args)