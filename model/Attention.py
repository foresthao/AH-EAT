'''
Adversarial Hierarchical-Aware Edge Attention Learning Method for Network Intrusion Detection
@mrforesthao
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from model.SAGE import SAGE

#调用
class HierarchicalGAT1(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, num_fine_classes):
        super().__init__()
        
        # 共享特征提取器
        self.gat = GAT(
            ndim_in=node_feat_dim,
            ndim_out=node_feat_dim,  # 扩大特征维度
            edim=edge_feat_dim,
            activation=F.leaky_relu,
            dropout=0.3
        )
        
        # 粗粒度分类头
        # self.coarse_head = nn.Sequential(
        #     nn.Linear(node_feat_dim*2, 32),  # MLPPredictor修改后的结构
        #     nn.ReLU(),
        #     nn.Linear(32, 2)
        # )
        self.coarse_head = MLPPredictor(
            in_features = node_feat_dim,
            out_classes = 2
        )
        
        # 细粒度分类头
        # self.fine_head = nn.Sequential(
        #     nn.Linear(node_feat_dim*2, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(64, num_fine_classes)
        # )
        self.fine_head = MLPPredictor(
            in_features = node_feat_dim,
            out_classes = num_fine_classes
        )

    def forward(self, g, nfeats, efeats):
        # 共享特征提取
        h = self.gat(g, nfeats, efeats)  # [E, 64]
        
        # 粗粒度分类
        coarse_pred = self.coarse_head(g, h)
        
        # 细粒度分类（动态掩码）
        fine_pred = self.fine_head(g, h)
        
        return coarse_pred, fine_pred


#层级化GAT模型，调用了GAT
class HierarchicalGAT(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, num_fine_classes):
        super().__init__()
        
        # 共享特征提取器
        self.gat = GAT(
            ndim_in=node_feat_dim,
            ndim_out=node_feat_dim,  # 扩大特征维度
            edim=edge_feat_dim,
            activation=F.leaky_relu,
            dropout=0.3
        )
        
        # 粗粒度分类头
        self.coarse_head = nn.Sequential(
            nn.Linear(node_feat_dim*2, 32),  # MLPPredictor修改后的结构
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # 细粒度分类头
        self.fine_head = nn.Sequential(
            nn.Linear(node_feat_dim*2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_fine_classes)
        )

    def forward(self, g, nfeats, efeats):
        # 共享特征提取
        h = self.gat(g, nfeats, efeats)  # [E, 64]
        
        # 获取边端点特征
        src, dst = g.edges()
        h_src = h[src]  # [E, 64]
        h_dst = h[dst]
        pair_feats = torch.cat([h_src, h_dst], dim=1)  # [E, 128]
        
        # 粗粒度分类
        coarse_logits = self.coarse_head(pair_feats)
        
        # 细粒度分类（动态掩码）
        fine_logits = self.fine_head(pair_feats)
        
        return coarse_logits, fine_logits

class GATLayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(GATLayer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation
        self.attn_fc = nn.Linear(2 * ndim_out, 1, bias=False)
        # self.attn_fc = nn.Linear(2 * ndim_in, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=2)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'m': edges.data['z'] * edges.data['a']}

    def reduce_func(self, nodes):
        return {'h_neigh': torch.sum(nodes.mailbox['m'], dim=1)}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats

            # 计算消息
            g.apply_edges(lambda edges: {'z': self.W_msg(torch.cat([edges.src['h'], edges.data['h']], 2))})

            # 计算注意力系数
            g.apply_edges(self.edge_attention)

            # 对注意力系数进行归一化
            g.edata['a'] = F.softmax(g.edata['e'], dim=1)

            # 消息传递和聚合
            g.update_all(self.message_func, self.reduce_func)

            # 更新节点特征
            g.ndata['h'] = self.activation(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))

            return g.ndata['h']


class GAT(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(ndim_in=ndim_in, edims=edim, ndim_out=128, activation=activation))
        self.layers.append(GATLayer(ndim_in=128, edims=edim, ndim_out=128, activation=activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

#没有使用注意力机制对特征融合的贡献，改用均值聚合
class RoNGAT_NA(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, num_fine_classes):
        super().__init__()
        
        # 共享特征提取器
        self.gat = SAGE(
            ndim_in=node_feat_dim,
            ndim_out=node_feat_dim,  # 扩大特征维度
            edim=edge_feat_dim,
            activation=F.leaky_relu,
            dropout=0.3
        )
        
        # 粗粒度分类头
        self.coarse_head = nn.Sequential(
            nn.Linear(node_feat_dim*2, 32),  # MLPPredictor修改后的结构
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # 细粒度分类头
        self.fine_head = nn.Sequential(
            nn.Linear(node_feat_dim*2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_fine_classes)
        )

    def forward(self, g, nfeats, efeats):
        # 共享特征提取
        h = self.gat(g, nfeats, efeats)  # [E, 64]
        
        # 获取边端点特征
        src, dst = g.edges()
        h_src = h[src]  # [E, 64]
        h_dst = h[dst]
        pair_feats = torch.cat([h_src, h_dst], dim=1)  # [E, 128]
        
        # 粗粒度分类
        coarse_logits = self.coarse_head(pair_feats)
        
        # 细粒度分类（动态掩码）
        fine_logits = self.fine_head(pair_feats)
        
        return coarse_logits, fine_logits

class RoNGAT_NH(nn.Module):
    """ 移除层级检测，直接多分类（合并粗+细粒度） """
    def __init__(self, node_feat_dim, edge_feat_dim, num_fine_classes):
        super().__init__()
        
        # 共享特征提取器（保留注意力）
        self.gat = GAT(
            ndim_in=node_feat_dim,
            ndim_out=node_feat_dim,
            edim=edge_feat_dim,
            activation=F.leaky_relu,
            dropout=0.3
        )
        
        # 合并分类头：输出为总类别数（假设良性类为0，细粒度攻击类为1~K）
        self.combined_head = MLPPredictor(
            in_features=node_feat_dim,
            out_classes=num_fine_classes  
        )

    def forward(self, g, nfeats, efeats):
        h = self.gat(g, nfeats, efeats)
        combined_pred = self.combined_head(g, h)
        
        # 为了接口兼容性，返回两个预测（第二个置零）
        dummy_coarse_pred = torch.zeros_like(combined_pred[:, :2])  # 虚拟粗粒度输出
        return dummy_coarse_pred, combined_pred