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
from model.Attention import HierarchicalGAT
from model.SAGE import HierarchicalSAGE, SAGE, MLPPredictor
from model.adversarial import PGDAdversary, SafePGDAdversary
from dgl.dataloading import MultiLayerNeighborSampler, EdgeDataLoader
from loss import AdaptiveHierarchicalLoss3, AdaptiveHierarchicalLoss4, AdaptiveHierarchicalLoss5
from sklearn.utils import class_weight

def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()

def compute_f1_score(pred, labels):
    pred_labels = pred.argmax(1).cpu().numpy()
    true_labels = labels.cpu().numpy()
    return f1_score(true_labels, pred_labels, average='weighted')

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

        # ================== 训练监控 ==================
        if epoch % 50 == 0:
            # 学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 粗粒度评估
            with torch.no_grad():
                coarse_acc = compute_accuracy(coarse_pred[train_mask], coarse_labels[train_mask])
                coarse_f1 = compute_f1_score(coarse_pred[train_mask], coarse_labels[train_mask])
                
                # 细粒度评估
                fine_acc = compute_accuracy(fine_pred[train_mask], fine_labels[train_mask])
                fine_f1 = compute_f1_score(fine_pred[train_mask], fine_labels[train_mask])

            
            # 打印监控信息
            print(f"Epoch {epoch:03d} | "
                  f"Loss: {current_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Coarse Acc: {coarse_acc:.4f} | Coarse F1: {coarse_f1:.4f} |"
                  f"Fine Acc: {fine_acc:.4f} | Fine F1: {fine_f1:.4f}")


def train1(model, G, device, args):
    #去掉的lr动态调整
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
        adv_G = pgd_adversary.generate(G, target='fine', epoch=epoch)
        
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

'''
不用adv_G, 不用adv, RoGAT-NADV训练
'''
def train2(model, G, device, args):
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
    # pgd_adversary = PGDAdversary(
    #     model=model,
    #     epsilon=0.05,  # 减小扰动幅度
    #     alpha=0.005,   # 减小步长
    #     iterations=5   # 减少迭代次数
    # )



    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()

        # ================== 对抗样本生成 ==================
        # adv_G = pgd_adversary.generate(G, target='alternate', epoch=epoch)
        
        # ================== 前向计算 ==================
        # 原始样本
        coarse_pred, fine_pred = model(G, G.ndata['h'], G.edata['h'])
        # 对抗样本
        # adv_coarse, adv_fine = model(adv_G, adv_G.ndata['h'], adv_G.edata['h'])
        
        # ================== 损失计算 ==================
        # 合并预测结果
        # combined_coarse = torch.cat([coarse_pred, adv_coarse], dim=0)
        # combined_fine = torch.cat([fine_pred, adv_fine], dim=0)
        # combined_labels_coarse = torch.cat([coarse_labels, coarse_labels], dim=0)
        # combined_labels_fine = torch.cat([fine_labels, fine_labels], dim=0)
        # combined_mask = torch.cat([train_mask, train_mask], dim=0)
        coarse_pred = torch.zeros(fine_pred.shape).to(device)
        
        loss, loss_dict = criterion(
            # combined_coarse[combined_mask],
            # combined_fine[combined_mask],
            # combined_labels_coarse[combined_mask],
            # combined_labels_fine[combined_mask]
            coarse_pred[train_mask],
            fine_pred[train_mask],
            coarse_labels[train_mask],
            fine_labels[train_mask]
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


'''
不用adv_G, 不用粗粒度loss, 不用Adaptiveloss
'''
def train3(model, G, device, args):
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
    
    # criterion = AdaptiveHierarchicalLoss5(
    #     max_epoch = args.num_epochs,
    #     coarse_class_weights=torch.tensor([1.0, 5.0]),
    #     fine_class_weights=fine_weights
    # ).to(device)
    criterion = nn.CrossEntropyLoss(weight=fine_weights)

    # 4. 数据准备
    train_mask = G.edata['train_mask']
    coarse_labels = G.edata['Label']
    fine_labels = G.edata['Attack']
    node_features = G.ndata['h']
    edge_features = G.edata['h']
    
    # 5. 初始化对抗生成器（更保守的参数）
    # pgd_adversary = PGDAdversary(
    #     model=model,
    #     epsilon=0.05,  # 减小扰动幅度
    #     alpha=0.005,   # 减小步长
    #     iterations=5   # 减少迭代次数
    # )



    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()

        # ================== 对抗样本生成 ==================
        # adv_G = pgd_adversary.generate(G, target='alternate', epoch=epoch)
        
        # ================== 前向计算 ==================
        # 原始样本
        coarse_pred, fine_pred = model(G, G.ndata['h'], G.edata['h'])
        # 对抗样本
        # adv_coarse, adv_fine = model(adv_G, adv_G.ndata['h'], adv_G.edata['h'])
        
        # ================== 损失计算 ==================
        # 合并预测结果
        # combined_coarse = torch.cat([coarse_pred, adv_coarse], dim=0)
        # combined_fine = torch.cat([fine_pred, adv_fine], dim=0)
        # combined_labels_coarse = torch.cat([coarse_labels, coarse_labels], dim=0)
        # combined_labels_fine = torch.cat([fine_labels, fine_labels], dim=0)
        # combined_mask = torch.cat([train_mask, train_mask], dim=0)
        coarse_pred = torch.zeros(fine_pred.shape).to(device)
        
        # loss, loss_dict = criterion(
        #     # combined_coarse[combined_mask],
        #     # combined_fine[combined_mask],
        #     # combined_labels_coarse[combined_mask],
        #     # combined_labels_fine[combined_mask]
        #     coarse_pred[train_mask],
        #     fine_pred[train_mask],
        #     coarse_labels[train_mask],
        #     fine_labels[train_mask]
        # )
        loss = criterion(fine_pred[train_mask], fine_labels[train_mask])
            
        loss.backward()
        optimizer.step()
        
        # ================== 训练监控 ==================
        if epoch % 50 == 0:
            # 学习率
            current_lr = optimizer.param_groups[0]['lr']
            current_loss = loss.item()
            # # 粗粒度评估
            # with torch.no_grad():
            #     # coarse_acc = (coarse_pred[train_mask].argmax(1) == coarse_labels[train_mask]).float().mean()
            #     coarse_acc = compute_accuracy(coarse_pred[train_mask], coarse_labels[train_mask])
            #     coarse_f1 = compute_f1_score(coarse_pred[train_mask], coarse_labels[train_mask])
                
            #     # 细粒度评估
            #     mask = (coarse_labels[train_mask] == 1)
            #     if mask.sum() > 0:
            #         # fine_acc = (fine_pred[train_mask][mask].argmax(1) == fine_labels[train_mask][mask]).float().mean()
            #         fine_acc = compute_accuracy(fine_pred[train_mask][mask], fine_labels[train_mask][mask])
            #         fine_f1 = compute_f1_score(fine_pred[train_mask][mask], fine_labels[train_mask][mask])
            #     else:
            #         fine_acc = torch.tensor(0.0)
            #         fine_f1 = torch.tensor(0.0)
            # mask = (coarse_labels[train_mask] == 1)
            fine_acc = compute_accuracy(fine_pred[train_mask], fine_labels[train_mask])
            fine_f1 = compute_f1_score(fine_pred[train_mask], fine_labels[train_mask])
            # 打印监控信息
            print(f"Epoch {epoch:03d} | "
                  f"Loss: {current_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                #   f"Coarse Acc: {coarse_acc:.4f} | Coarse F1: {coarse_f1:.4f} |"
                  f"Fine Acc: {fine_acc:.4f} | Fine F1: {fine_f1:.4f}")