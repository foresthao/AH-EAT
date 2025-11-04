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
from loss import AdaptiveHierarchicalLoss2, AdaptiveHierarchicalLoss3, AdaptiveHierarchicalLoss4, AdaptiveHierarchicalLoss5
from dgl.dataloading import MultiLayerNeighborSampler, EdgeDataLoader

def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()

def compute_f1_score(pred, labels):
    pred_labels = pred.argmax(1).cpu().numpy()
    true_labels = labels.cpu().numpy()
    return f1_score(true_labels, pred_labels, average='weighted')

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

        # 保存预测结果
        torch.save(coarse_pred.cpu(), 'coarse_pred.pt')
        torch.save(fine_pred.cpu(), 'fine_pred.pt')

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

def evaluate_confusion(model, G_test, device, args):
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

        # 保存预测结果
        torch.save(coarse_pred.cpu(), f'./dataset/{args.dataset}_coarse_pred2.pt')
        torch.save(fine_pred.cpu(), f'./dataset/{args.dataset}_fine_pred2.pt')

        # 粗粒度准确率
        coarse_labels = G_test.edata['Label'][test_mask]
        torch.save(coarse_labels.cpu(), f'./dataset/{args.dataset}_coarse_labels2.pt')
        coarse_acc = compute_accuracy(coarse_pred[test_mask], coarse_labels)
        coarse_f1 = compute_f1_score(coarse_pred[test_mask], coarse_labels)

        # 细粒度准确率（仅在粗粒度预测为攻击时计算）
        fine_labels = G_test.edata['Attack'][test_mask]
        # 保存细粒度真实标签
        torch.save(fine_labels.cpu(), f'./dataset/{args.dataset}_fine_labels2.pt')
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

def evaluate_adv1(model, G_test, device, adversary, target='coarse', epoch=None):
    '''在测试集上评估模型性能，计算Clean Accuracy, Robust Accuracy, ASR和Confidence Shift'''
    model.eval()

    # 计算Clean Accuracy
    with torch.no_grad():
        # 获取测试掩码（测试图的所有边均为测试样本）
        test_mask = G_test.edata.get(
            'test_mask', 
            torch.ones(G_test.num_edges(), dtype=torch.bool, device=device)
        )

        # 向前传播
        coarse_pred_clean, fine_pred_clean = model(G_test, G_test.ndata['h'], G_test.edata['h'])

        # 粗粒度准确率
        coarse_labels = G_test.edata['Label'][test_mask]
        coarse_acc_clean = compute_accuracy(coarse_pred_clean[test_mask], coarse_labels)
        coarse_f1_clean = compute_f1_score(coarse_pred_clean[test_mask], coarse_labels)

        # 细粒度准确率（仅在粗粒度预测为攻击时计算）
        fine_labels = G_test.edata['Attack'][test_mask]
        fine_acc_clean = compute_accuracy(fine_pred_clean[test_mask], fine_labels)
        fine_f1_clean = compute_f1_score(fine_pred_clean[test_mask], fine_labels)

    # 生成对抗样本
    adv_G = adversary.generate(G_test, target=target, epoch=epoch)

    with torch.no_grad():
        # 获取测试掩码（测试图的所有边均为测试样本）
        test_mask = adv_G.edata.get(
            'test_mask', 
            torch.ones(adv_G.num_edges(), dtype=torch.bool, device=device)
        )

        # 向前传播
        coarse_pred_adv, fine_pred_adv = model(adv_G, adv_G.ndata['h'], adv_G.edata['h'])

        # 粗粒度准确率
        coarse_labels = G_test.edata['Label'][test_mask]
        coarse_acc_adv = compute_accuracy(coarse_pred_adv[test_mask], coarse_labels)
        coarse_f1_adv = compute_f1_score(coarse_pred_adv[test_mask], coarse_labels)

        # 细粒度准确率（仅在粗粒度预测为攻击时计算）
        fine_labels = G_test.edata['Attack'][test_mask]
        fine_acc_adv = compute_accuracy(fine_pred_adv[test_mask], fine_labels)
        fine_f1_adv = compute_f1_score(fine_pred_adv[test_mask], fine_labels)

        # 计算ASR
        coarse_asr = 1 - coarse_acc_adv
        fine_asr = 1 - fine_acc_adv

        # 计算Confidence Shift
        coarse_confidence_clean = torch.softmax(coarse_pred_clean[test_mask], dim=1).max(dim=1)[0]
        coarse_confidence_adv = torch.softmax(coarse_pred_adv[test_mask], dim=1).max(dim=1)[0]
        coarse_confidence_shift = (coarse_confidence_clean - coarse_confidence_adv).mean().item()

        fine_confidence_clean = torch.softmax(fine_pred_clean[test_mask], dim=1).max(dim=1)[0]
        fine_confidence_adv = torch.softmax(fine_pred_adv[test_mask], dim=1).max(dim=1)[0]
        fine_confidence_shift = (fine_confidence_clean - fine_confidence_adv).mean().item()

    return (coarse_acc_clean, coarse_f1_clean, fine_acc_clean, fine_f1_clean,
            coarse_acc_adv, coarse_f1_adv, fine_acc_adv, fine_f1_adv,
            coarse_asr, fine_asr, coarse_confidence_shift, fine_confidence_shift)


def evaluate_adv2(model, G_test, device, adversary, target='fine', epoch=None):
    '''在测试集上评估模型性能，计算Clean Accuracy, Robust Accuracy, ASR'''
    model.eval()

    # 计算Clean Accuracy
    with torch.no_grad():
        # 获取测试掩码（测试图的所有边均为测试样本）
        test_mask = G_test.edata.get(
            'test_mask', 
            torch.ones(G_test.num_edges(), dtype=torch.bool, device=device)
        )

        # 向前传播
        coarse_pred_clean, fine_pred_clean = model(G_test, G_test.ndata['h'], G_test.edata['h'])

        # 粗粒度准确率
        coarse_labels = G_test.edata['Label'][test_mask]
        coarse_acc_clean = compute_accuracy(coarse_pred_clean[test_mask], coarse_labels)
        coarse_f1_clean = compute_f1_score(coarse_pred_clean[test_mask], coarse_labels)

        # 细粒度准确率（仅在粗粒度预测为攻击时计算）
        fine_labels = G_test.edata['Attack'][test_mask]
        fine_acc_clean = compute_accuracy(fine_pred_clean[test_mask], fine_labels)
        fine_f1_clean = compute_f1_score(fine_pred_clean[test_mask], fine_labels)

    # 生成对抗样本
    adv_G = adversary.generate(G_test, target=target, epoch=epoch)

    with torch.no_grad():
        # 获取测试掩码（测试图的所有边均为测试样本）
        test_mask = adv_G.edata.get(
            'test_mask', 
            torch.ones(adv_G.num_edges(), dtype=torch.bool, device=device)
        )

        # 向前传播
        coarse_pred_adv, fine_pred_adv = model(adv_G, adv_G.ndata['h'], adv_G.edata['h'])

        # 粗粒度准确率
        coarse_labels = G_test.edata['Label'][test_mask]
        coarse_acc_adv = compute_accuracy(coarse_pred_adv[test_mask], coarse_labels)
        coarse_f1_adv = compute_f1_score(coarse_pred_adv[test_mask], coarse_labels)

        # 细粒度准确率（仅在粗粒度预测为攻击时计算）
        fine_labels = G_test.edata['Attack'][test_mask]
        fine_acc_adv = compute_accuracy(fine_pred_adv[test_mask], fine_labels)
        fine_f1_adv = compute_f1_score(fine_pred_adv[test_mask], fine_labels)

        # 计算ASR
        coarse_asr = 1 - coarse_acc_adv
        fine_asr = 1 - fine_acc_adv


    return (coarse_acc_clean, coarse_f1_clean, fine_acc_clean, fine_f1_clean,
            coarse_acc_adv, coarse_f1_adv, fine_acc_adv, fine_f1_adv,
            coarse_asr, fine_asr)

def evaluate_embedding_tsne(model, G_test, device, args):
    '''在测试集上评估模型性能并保存embedding数据用于t-SNE可视化'''
    model.eval()
    with torch.no_grad():
        # 获取测试掩码（测试图的所有边均为测试样本）
        test_mask = G_test.edata.get(
            'test_mask', 
            torch.ones(G_test.num_edges(), dtype=torch.bool, device=device)
        )

        # 向前传播获取embedding
        # 首先获取GAT层的embedding（节点级别）
        h = model.gat(G_test, G_test.ndata['h'], G_test.edata['h'])  # [N, feat_dim]
        
        # 获取预测结果
        coarse_pred, fine_pred = model(G_test, G_test.ndata['h'], G_test.edata['h'])

        # 将节点特征转换为边特征（通过边的源节点和目标节点）
        src, dst = G_test.edges()
        h_src = h[src]  # [E, feat_dim] - 源节点特征
        h_dst = h[dst]  # [E, feat_dim] - 目标节点特征
        
        # 拼接源节点和目标节点特征作为边的embedding
        ah_eat_embedding = torch.cat([h_src, h_dst], dim=1)  # [E, feat_dim*2]
        
        # 应用测试掩码
        ah_eat_embedding = ah_eat_embedding[test_mask].cpu().numpy()  # [样本数量, 特征维度*2]
        
        # 保存标签数据
        coarse_labels = G_test.edata['Label'][test_mask].cpu().numpy()  # [样本数量,]
        fine_labels = G_test.edata['Attack'][test_mask].cpu().numpy()   # [样本数量,]
        
        # 保存到本地文件
        save_dir = f'./embeddings/{args.dataset}'
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存embedding和标签
        np.save(f'{save_dir}/ah_eat_embedding.npy', ah_eat_embedding)
        np.save(f'{save_dir}/coarse_labels.npy', coarse_labels)
        np.save(f'{save_dir}/fine_labels.npy', fine_labels)
        
        # 同时保存预测结果（保持与原函数兼容）
        torch.save(coarse_pred.cpu(), f'./dataset/{args.dataset}_coarse_pred.pt')
        torch.save(fine_pred.cpu(), f'./dataset/{args.dataset}_fine_pred.pt')
        torch.save(coarse_labels, f'./dataset/{args.dataset}_coarse_labels.pt')
        torch.save(fine_labels, f'./dataset/{args.dataset}_fine_labels.pt')

        # 计算性能指标
        coarse_acc = compute_accuracy(coarse_pred[test_mask], G_test.edata['Label'][test_mask])
        coarse_f1 = compute_f1_score(coarse_pred[test_mask], G_test.edata['Label'][test_mask])
        fine_acc = compute_accuracy(fine_pred[test_mask], G_test.edata['Attack'][test_mask])
        fine_f1 = compute_f1_score(fine_pred[test_mask], G_test.edata['Attack'][test_mask])

        print(f"Embedding shape: {ah_eat_embedding.shape}")
        print(f"Coarse labels shape: {coarse_labels.shape}")
        print(f"Fine labels shape: {fine_labels.shape}")
        print(f"Embedding data saved to: {save_dir}")

        return coarse_acc, coarse_f1, fine_acc, fine_f1