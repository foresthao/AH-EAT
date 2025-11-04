'''
Adversarial Hierarchical-Aware Edge Attention Learning Method for Network Intrusion Detection
@mrforesthao
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl.function as fn

#层级化SAGE模型，调用了SAGE
class HierarchicalSAGE(nn.Module):
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

class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        ### force to outut fix dimensions
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)#线性层，用于拼接后的节点特征和边特征
        ### apply weight
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)#线性层，用于拼接进一步处理节点特征和聚合后的邻居节点特征
        self.activation = activation

    def message_func(self, edges):
        return {'m': self.W_msg(torch.cat([edges.src['h'], edges.data['h']], 2))}#拼接原节点特征和边特征，在维度2拼接后进行变换

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            # Eq4
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))#使用 g.update_all 进行消息传递和聚合，self.message_func 生成消息，fn.mean('m', 'h_neigh') 对消息进行平均聚合，并将结果存储为 h_neigh。
            # Eq5          
            g.ndata['h'] = F.relu(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            return g.ndata['h']


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        self.layers.append(SAGELayer(128, edim, ndim_out, activation))
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

class KANPredictor(nn.Module):
    def __init__(self, in_features, out_classes, hidden_dim=64):
        super().__init__()
        # 知识感知层
        self.knowledge_layer = nn.Sequential(
            nn.Linear(in_features * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 关系转换层
        self.relation_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, out_classes)
        
    def apply_edges(self, edges):
        # 获取源节点和目标节点的特征
        h_u = edges.src['h']
        h_v = edges.dst['h']
        
        # 拼接节点特征
        pair_feats = torch.cat([h_u, h_v], 1)
        
        # 知识感知处理
        x = self.knowledge_layer(pair_feats)
        
        # 关系转换
        x = self.relation_transform(x)
        
        # 生成预测分数
        score = self.output_layer(x)
        return {'score': score}
    
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


