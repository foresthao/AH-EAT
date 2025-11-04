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
from model.Attention import HierarchicalGAT1, HierarchicalGAT, RoNGAT_NA, RoNGAT_NH
from model.Adversarial import PGDAdversary, SafePGDAdversary
from loss import AdaptiveHierarchicalLoss, AdaptiveHierarchicalLoss2, AdaptiveHierarchicalLoss3, AdaptiveHierarchicalLoss4, AdaptiveHierarchicalLoss5
from train import train1, train2
from evaluation import evaluate, evaluate_adv1, evaluate_adv2
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
    if args.ablation == 'RoNGAT-NA':
        model = RoNGAT_NA(node_feat_dim=G.ndata['h'].shape[2],
                        edge_feat_dim=G.edata['h'].shape[2],
                        num_fine_classes=num_fine_class).to(device)
    elif args.ablation == 'RoNGAT-NH':
        model = RoNGAT_NH(node_feat_dim=G.ndata['h'].shape[2],
                        edge_feat_dim=G.edata['h'].shape[2],
                        num_fine_classes=num_fine_class).to(device)
    elif args.ablation == 'RoNGAT-NADV':
        model = HierarchicalGAT(node_feat_dim=G.ndata['h'].shape[2],
                            edge_feat_dim=G.edata['h'].shape[2],
                            num_fine_classes=num_fine_class).to(device)#RoNGAT_ADV
    else:
        print('error model')

    print('2.开始训练')
    # 调用train方法
    if args.ablation == 'RoNGAT-NA' or 'RoNGAT-NH':
        train1(model=model, G=G, device=device, args=args)#RoGAR-NA, NH
    elif args.ablation == 'RoNGAT-NADV':
        train2(model=model, G=G, device=device, args=args)#train2是RoGAT-NADV使用的
    else:
        print('error ablation study method')

    adversary = PGDAdversary(
        model=model,
        epsilon=0.05,  # 扰动幅度
        alpha=0.005,   # 步长
        iterations=10   # 迭代次数
    )

    print('3.在测试集上评估最终模型')
    # test_coarse_acc, test_coarse_f1, test_fine_acc, test_fine_f1 = evaluate(model, G_test, device)
    # test_coarse_acc, test_coarse_f1, test_fine_acc, test_fine_f1 = evaluate_adv(model, G_test, device, adversary, target='fine')
    (coarse_acc_clean, coarse_f1_clean, fine_acc_clean, fine_f1_clean, coarse_acc_adv, coarse_f1_adv, fine_acc_adv, fine_f1_adv, coarse_asr, fine_asr) = evaluate_adv2(model, G_test, device, adversary)
    print(
        f'[Final Test] 粗粒度准确率: {coarse_acc_clean:.4f}, 粗粒度f1:{coarse_f1_clean:.4f}, 细粒度准确率: {fine_acc_clean:.4f}, 细粒度f1:{fine_f1_clean:.4f}||'
        f'[Adversal] 粗粒度准确率: {coarse_acc_adv:.4f}, 粗粒度f1:{coarse_f1_adv:.4f}, 细粒度准确率: {fine_acc_adv:.4f}, 细粒度f1:{fine_f1_adv:.4f}||'
        f'[asr&confidence] 粗粒度asr:{coarse_asr:.4f}, 细粒度asr:{fine_asr:.4f}'
    )

if __name__ == '__main__':
    #1.导入数据，分别是具有粗细两种粒度的图
    parser = argparse.ArgumentParser(description='ablation study')
    parser.add_argument('--dataset', type=str, default='BoT-IoT', choices=['BoT-IoT', 'ToN-IoT', 'CIC-IDS2018', 'UNSW-NB15'])
    parser.add_argument("--num_runs", type=int, default=1, help="number of runs, default = 10")
    parser.add_argument("--num_epochs", type=int, default=2001, help="number of training epochs, default = 50")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0, help="weight decay")
    parser.add_argument('--gnn', type=str, default='GAT', help="Graph attention network")
    parser.add_argument('--batch_size', type=int, default=1024, help="batch size for training")
    parser.add_argument('--ablation', type=str, default='RoNGAT-NA', choices=['RoNGAT-NA', 'RoNGAT-NH', 'RoNGAT-NADV'], help="Ablation Study")

    args = parser.parse_args()
    
    print(args)
    run(args)
