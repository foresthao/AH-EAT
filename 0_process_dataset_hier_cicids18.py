'''
Adversarial Hierarchical-Aware Edge Attention Learning Method for Network Intrusion Detection
@mrforesthao
'''
import os
import pickle
import dgl.nn as dglnn
from dgl import from_networkx
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl.function as fn
import networkx as nx
import pandas as pd
import socket
import struct
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import compute_accuracy

'''
适用数据集NF-CSE-CIC-IDS2018-v2
'''

print('开始进行数据的预处理')
print('1.读入数据')
dataset_path = '/data2/yanhao_data/e_gat/dataset/'
data = pd.read_csv('/data2/yanhao_data/e_gat/dataset/NF-CSE-CIC-IDS2018-v2.csv')

data['IPV4_SRC_ADDR'] = data.IPV4_SRC_ADDR.apply(str)
data['IPV4_DST_ADDR'] = data.IPV4_DST_ADDR.apply(str)

data.drop(columns=['L4_SRC_PORT','L4_DST_PORT'],inplace=True)

le = LabelEncoder()
le.fit_transform(data.Attack.values)
data['Attack'] = le.transform(data['Attack'])

print(data.head())

#进行采样15%，只有CIC-IDS数据集的时候才进行采样
data = data.groupby(by='Attack').sample(frac=0.1, random_state=13)
print(data.groupby(by="Attack").count())

label = data['Label']
attack = data['Attack']
data.drop(columns=['Attack'],inplace = True)
data.drop(columns=['Label'],inplace = True)

print('2.进行转换')
data =  pd.concat([data, label, attack], axis=1)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=123,stratify= label)

#把无限大和无限小的数据进行替换
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

encoder = ce.TargetEncoder(cols=['TCP_FLAGS','L7_PROTO','PROTOCOL',
                                    'CLIENT_TCP_FLAGS','SERVER_TCP_FLAGS','ICMP_TYPE',
                                  'ICMP_IPV4_TYPE','DNS_QUERY_ID','DNS_QUERY_TYPE',
                                  'FTP_COMMAND_RET_CODE'])#目标编码器类，用于对分类特征进行编码
encoder.fit(X_train, y_train)
X_train = encoder.transform(X_train)

print(X_train.head())
cols_to_norm = list(set(list(X_train.iloc[:, 2:-2].columns)))

# scaler = StandardScaler()#标准化数据，把数值转化成均值为0，方差为1的分布
scaler = Normalizer()
X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])
X_train['h'] = X_train[ cols_to_norm ].values.tolist()

print('3.构图')
G = nx.from_pandas_edgelist(X_train, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h','Label','Attack'],create_using=nx.MultiGraph())
G = G.to_directed()
G = from_networkx(G,edge_attrs=['h','Label', 'Attack'] )
G.ndata['h'] = th.ones(G.num_nodes(), G.edata['h'].shape[1])
G.edata['train_mask'] = th.ones(len(G.edata['h']), dtype=th.bool)
print(G.edata['train_mask'])


X_test = encoder.transform(X_test)
X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])
X_test['h'] = X_test[ cols_to_norm ].values.tolist()

G_test = nx.from_pandas_edgelist(X_test, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", ['h','Label', 'Attack'],create_using=nx.MultiGraph())
G_test = G_test.to_directed()
G_test = from_networkx(G_test,edge_attrs=['h','Label', 'Attack'] )
G_test.ndata['h'] = th.ones(G_test.num_nodes(), G.ndata['h'].shape[1])


print('4.保存')
# 使用 pickle 保存图
save_path = os.path.join(dataset_path, "graph/","saved_graph_cic_ids18_10%.pkl")
save_path_test = os.path.join(dataset_path, "graph/","saved_graph_cic_ids18_10%_test.pkl")
with open(save_path, 'wb') as f:
    pickle.dump(G, f)
print(f"图已使用 pickle 保存到 {save_path}")

with open(save_path_test, 'wb') as f:
    pickle.dump(G_test, f)
print(f"测试图已使用 pickle 保存到 {save_path_test}")

# 使用 pickle 加载图
# with open(save_path, 'rb') as f:
#     loaded_G = pickle.load(f)
# print("使用 pickle 加载的图：", loaded_G)