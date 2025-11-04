'''
Adversarial Hierarchical-Aware Edge Attention Learning Method for Network Intrusion Detection
@mrforesthao
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import os

def make_meshgrid(x, y, h=.02):
    """创建用于绘制决策边界的网格"""
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """绘制分类器的决策边界"""
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def visualize_tsne(dataset_name, label_type='coarse', n_samples=None, perplexity=30):
    """使用t-SNE可视化embedding数据"""
    save_dir = f'./embeddings/{dataset_name}'
    
    # 加载数据
    embedding_file = f'{save_dir}/ah_eat_embedding.npy'
    coarse_labels_file = f'{save_dir}/coarse_labels.npy'
    fine_labels_file = f'{save_dir}/fine_labels.npy'
    
    if not os.path.exists(embedding_file):
        print(f"Embedding file not found: {embedding_file}")
        return
    
    # 加载embedding和标签
    ah_eat_embedding = np.load(embedding_file)
    coarse_labels = np.load(coarse_labels_file)
    fine_labels = np.load(fine_labels_file)
    
    # 选择标签类型
    if label_type == 'coarse':
        labels = coarse_labels
        label_names = ['Benign', 'Attack']
        title_suffix = 'Coarse-grained'
    else:
        labels = fine_labels
        # 根据数据集设置细粒度标签名称
        if dataset_name == 'BoT-IoT':
            label_names = ['Benign', 'DoS', 'DDoS', 'Reconnaissance', 'Theft']
        elif dataset_name == 'CIC-IDS2018':
            label_names = ['Benign', 'BruteForce', 'Bot', 'DoS', 'DDoS', 'Infiltration', 'Web attack']
        else:
            label_names = [f'Class_{i}' for i in range(len(np.unique(labels)))]
        title_suffix = 'Fine-grained'
    
    # 如果指定了样本数量，随机采样
    if n_samples and n_samples < len(ah_eat_embedding):
        indices = np.random.choice(len(ah_eat_embedding), n_samples, replace=False)
        ah_eat_embedding = ah_eat_embedding[indices]
        labels = labels[indices]
    
    print(f"Visualizing {len(ah_eat_embedding)} samples for {dataset_name} ({title_suffix})")
    
    # 执行t-SNE降维
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    embedding_2d = tsne.fit_transform(ah_eat_embedding)
    
    # 创建可视化
    plt.figure(figsize=(12, 8))
    
    # 如果是粗粒度标签，则计算并绘制决策边界
    if label_type == 'coarse':
        # 训练KNN分类器用于决策边界
        print("Computing decision boundaries...")
        clf = KNeighborsClassifier(n_neighbors=255)#原来是5，25
        clf.fit(embedding_2d, labels)
        
        # 创建网格
        xx, yy = make_meshgrid(embedding_2d[:,0], embedding_2d[:,1], h=0.2)#原来是0.1
        
        # 绘制决策边界
        plot_contours(plt.gca(), clf, xx, yy, 
                    cmap=plt.cm.Pastel1, alpha=0.3)
    
    # 为每个类别绘制不同颜色的点
    unique_labels = np.unique(labels)
    # 使用一组鲜明、区分度高的颜色 (Tableau 20)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', 
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_color = colors[i % len(colors)]
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                   color=class_color, label=label_names[label] if label < len(label_names) else f'Class_{label}',
                   alpha=0.8, s=40)
    
    plt.title(f't-SNE Visualization of AH-EAT Embeddings\n{dataset_name} - {title_suffix}', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 保存图片
    save_path = f'./embeddings/{dataset_name}/tsne_{label_type}3.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    save_path = f'./embeddings/{dataset_name}/tsne_{label_type}3.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE visualization saved to: {save_path}")
    
    plt.show()

def visualize_all_datasets():
    """为所有数据集生成t-SNE可视化"""
    datasets = ['BoT-IoT', 'ToN-IoT', 'CIC-IDS2018', 'UNSW-NB15']
    
    for dataset in datasets:
        try:
            print(f"\n{'='*60}")
            print(f"Processing {dataset}")
            print(f"{'='*60}")
            
            # 粗粒度可视化
            visualize_tsne(dataset, label_type='coarse', n_samples=5000)
            
            # 细粒度可视化
            visualize_tsne(dataset, label_type='fine', n_samples=5000)
            
        except Exception as e:
            print(f"Error processing {dataset}: {e}")

if __name__ == '__main__':
    # 示例：可视化单个数据集
    visualize_tsne('ToN-IoT', label_type='fine', n_samples=30000)
    # visualize_tsne('BoT-IoT', label_type='fine', n_samples=30000)
    
    # 或者可视化所有数据集
    # visualize_all_datasets() 