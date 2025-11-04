# Adversarial Hierarchical-Aware Edge Attention Learning Method for Network Intrusion Detection

[![Published](https://img.shields.io/badge/Published-Appl._Sci._2025-blue)](https://doi.org/10.3390/app15147915)

This repository contains the implementation of the **Adversarial Hierarchical-Aware Edge Attention Learning (AH-EAT)** method for network intrusion detection, published in Applied Sciences 2025.

## 📖 项目简介

本项目提出了一种基于对抗性层次感知边注意力学习的网络入侵检测方法（AH-EAT）。该方法结合了图神经网络、注意力机制和对抗训练技术，实现了一个层次化的入侵检测框架，能够同时进行粗粒度（攻击/正常）和细粒度（具体攻击类型）的流量分类。

## ✨ 主要贡献

- **边特征与拓扑模式融合**：同时利用边特征和图的拓扑结构模式来检测攻击模式
- **注意力机制**：引入注意力机制，根据节点之间的重要性来聚合邻居节点特征，以更好地捕捉图的结构信息
- **层次化检测框架**：同时检测攻击流量的粗粒度（攻击/正常）和细粒度（具体攻击类别）分类，传统方法仅针对攻击和良性流量进行检测，本方法更多关注每种攻击类别的检测
- **对抗性增强**：使用对抗性增强方法，解决IDS领域对抗攻击逃逸问题，提升模型鲁棒性。采用PGD（Projected Gradient Descent）生成对抗性边特征扰动
- **基准测试验证**：在多个基准数据集上进行实验，验证了方法的有效性

## 🚀 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.8+
- DGL (Deep Graph Library)
- 其他依赖包：numpy, sklearn, matplotlib等

### 安装依赖

```bash
pip install torch dgl numpy scikit-learn matplotlib pandas networkx category-encoders seaborn
```

### 数据准备

1. 下载数据集（BoT-IoT, ToN-IoT, CIC-IDS2018, UNSW-NB15）
2. 使用预处理脚本处理数据：
   ```bash
   python 0_process_dataset_hier_cicids18.py  # 处理CIC-IDS2018数据集
   python 0_process_dataset_hier2.py          # 处理其他数据集
   python 0_process_dataset_hier3.py
   ```
3. 处理后的数据将保存在 `dataset/` 目录下

### 运行主程序

#### 训练和评估主模型（对抗性层次化模型）

```bash
# 使用BoT-IoT数据集
python 1_main_hierarchical_adv3.py --dataset BoT-IoT --num_epochs 2001 --lr 0.01 --batch_size 1024

# 使用CIC-IDS2018数据集
python 1_main_hierarchical_adv3.py --dataset CIC-IDS2018 --num_epochs 2001 --lr 0.01 --batch_size 1024

# 使用ToN-IoT数据集
python 1_main_hierarchical_adv3.py --dataset ToN-IoT --num_epochs 2001 --lr 0.01 --batch_size 1024

# 使用UNSW-NB15数据集
python 1_main_hierarchical_adv3.py --dataset UNSW-NB15 --num_epochs 2001 --lr 0.01 --batch_size 1024
```

#### 运行消融实验

```bash
python 2_main_ablation.py --dataset BoT-IoT
```

### 参数说明

- `--dataset`: 数据集选择，可选值：`BoT-IoT`, `ToN-IoT`, `CIC-IDS2018`, `UNSW-NB15`
- `--num_epochs`: 训练轮数，默认2001
- `--lr`: 学习率，默认0.01
- `--batch_size`: 批次大小，默认1024
- `--num_runs`: 运行次数，默认1

## 📁 项目结构

```
.
├── 0_process_dataset_hier*.py          # 数据集预处理脚本
│   ├── 0_process_dataset_hier_cicids18.py  # CIC-IDS2018数据集预处理
│   ├── 0_process_dataset_hier2.py          # 其他数据集预处理
│   └── 0_process_dataset_hier3.py           # 其他数据集预处理
│
├── 1_main_hierarchical_adv*.py         # 主训练脚本（对抗性层次化模型）
│   ├── 1_main_hierarchical_adv3.py          # 主要使用的训练脚本
│   └── 1_main_hierarchical_adv4.py          # 其他版本
│
├── 1_main_hierarchical*.py             # 其他层次化模型训练脚本
│   ├── 1_main_hierarchical1.py              # 基础层次化模型
│   └── 1_main_hierarchical_eva_rob.py       # 鲁棒性评估
│
├── 2_main_ablation.py                  # 消融实验脚本
│
├── model/                              # 模型定义
│   ├── Attention.py                        # 层次化GAT注意力模型
│   ├── adversarial.py                      # 对抗训练模块（PGD）
│   └── SAGE.py                             # GraphSAGE基础模块
│
├── dataset/                            # 数据集目录
│   ├── BoT-IoT_*.pt                        # BoT-IoT数据集处理后的文件
│   ├── ToN-IoT_*.pt                        # ToN-IoT数据集处理后的文件
│   ├── CIC-IDS2018_*.pt                   # CIC-IDS2018数据集处理后的文件
│   └── UNSW-NB15_*.pt                     # UNSW-NB15数据集处理后的文件
│
├── draw_fig/                           # 可视化脚本和结果
│   ├── confusion_matrix.py                  # 混淆矩阵绘制
│   ├── plot_fp_fn_analysis.py              # 假阳性/假阴性分析
│   └── *.png, *.pdf                        # 生成的图表
│
├── embeddings/                         # Embedding数据保存目录
│   └── {dataset_name}/                     # 各数据集的embedding数据
│       ├── ah_eat_embedding.npy
│       ├── coarse_labels.npy
│       └── fine_labels.npy
│
│
├── evaluation.py                       # 评估函数
├── loss.py                            # 损失函数定义
├── train.py                           # 训练函数
├── utils.py                           # 工具函数
├── visualization.py                   # 可视化函数
├── visualize_tsne.py                  # t-SNE可视化
│
└── README.md                          # 本文件
```

### 主要文件夹说明

- **`model/`**: 包含核心模型定义
  - `Attention.py`: 实现了层次化图注意力网络（HierarchicalGAT），包含粗粒度和细粒度分类头
  - `adversarial.py`: 实现了PGD对抗训练模块，用于提升模型鲁棒性
  - `SAGE.py`: GraphSAGE基础模块

- **`dataset/`**: 存储预处理后的数据集文件（.pt格式），包含训练和测试数据的图结构、节点特征、边特征和标签

- **`draw_fig/`**: 包含可视化脚本和生成的图表，用于结果分析和论文展示

- **`embeddings/`**: 保存模型生成的embedding向量，用于后续的t-SNE可视化分析

## 📊 实验结果

模型在多个基准数据集上进行了评估，包括：
- **BoT-IoT**
- **ToN-IoT**
- **CIC-IDS2018**
- **UNSW-NB15**

实验结果展示了本方法在粗粒度和细粒度分类任务上的优越性能。

## 📝 引用

如果您在研究中使用了本代码，请引用以下论文：

```bibtex
@article{ah_eat_2025,
  title={Adversarial Hierarchical-Aware Edge Attention Learning Method for Network Intrusion Detection},
  journal={Applied Sciences},
  volume={15},
  number={14},
  pages={7915},
  year={2025},
  doi={10.3390/app15147915},
  url={https://doi.org/10.3390/app15147915}
}
```

**引用格式（中文）**：
如果您的研究中使用中文引用格式，可以使用：
```
Adversarial Hierarchical-Aware Edge Attention Learning Method for Network Intrusion Detection. 
Appl. Sci. 2025, 15(14), 7915; https://doi.org/10.3390/app15147915
Submission received: 3 June 2025 / Revised: 6 July 2025 / Accepted: 12 July 2025 / Published: 16 July 2025
```

## 📄 论文信息

- **期刊**: Applied Sciences
- **卷号**: 15
- **期号**: 14
- **页码**: 7915
- **DOI**: https://doi.org/10.3390/app15147915
- **提交日期**: 2025年6月3日
- **修订日期**: 2025年7月6日
- **接受日期**: 2025年7月12日
- **发表日期**: 2025年7月16日

## 🔧 注意事项

1. **GPU要求**: 建议使用GPU进行训练，CPU训练速度较慢
2. **内存要求**: 处理大型数据集时可能需要较大的内存空间
3. **数据格式**: 数据集需要预处理成DGL图格式（.pt文件）
4. **版本兼容性**: 请确保PyTorch和DGL版本兼容

## 📧 联系方式

如有问题或建议，请通过GitHub Issues联系。或者yanhaoforest@gmail.com

## 📜 许可证

本项目遵循相应的开源许可证（请根据实际情况添加）。
