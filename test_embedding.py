'''
Adversarial Hierarchical-Aware Edge Attention Learning Method for Network Intrusion Detection
@mrforesthao
'''
import numpy as np
import os

def test_embedding_data(dataset_name):
    """测试保存的embedding数据"""
    save_dir = f'./embeddings/{dataset_name}'
    
    # 检查文件是否存在
    embedding_file = f'{save_dir}/ah_eat_embedding.npy'
    coarse_labels_file = f'{save_dir}/coarse_labels.npy'
    fine_labels_file = f'{save_dir}/fine_labels.npy'
    
    if not os.path.exists(embedding_file):
        print(f"Embedding file not found: {embedding_file}")
        return
    
    # 加载数据
    ah_eat_embedding = np.load(embedding_file)
    coarse_labels = np.load(coarse_labels_file)
    fine_labels = np.load(fine_labels_file)
    
    # 打印数据信息
    print(f"=== {dataset_name} Dataset Embedding Information ===")
    print(f"Embedding shape: {ah_eat_embedding.shape}")
    print(f"Coarse labels shape: {coarse_labels.shape}")
    print(f"Fine labels shape: {fine_labels.shape}")
    
    print(f"\nEmbedding statistics:")
    print(f"  Min: {ah_eat_embedding.min():.4f}")
    print(f"  Max: {ah_eat_embedding.max():.4f}")
    print(f"  Mean: {ah_eat_embedding.mean():.4f}")
    print(f"  Std: {ah_eat_embedding.std():.4f}")
    
    print(f"\nLabel distributions:")
    print(f"  Coarse labels: {np.bincount(coarse_labels)}")
    print(f"  Fine labels: {np.bincount(fine_labels)}")
    
    print(f"\nUnique labels:")
    print(f"  Coarse unique: {np.unique(coarse_labels)}")
    print(f"  Fine unique: {np.unique(fine_labels)}")
    
    # 验证数据一致性
    assert len(ah_eat_embedding) == len(coarse_labels) == len(fine_labels), "数据长度不一致！"
    print(f"\n✓ 数据长度一致性验证通过: {len(ah_eat_embedding)} 个样本")

if __name__ == '__main__':
    # 测试所有数据集
    datasets = ['BoT-IoT', 'ToN-IoT', 'CIC-IDS2018', 'UNSW-NB15']
    
    for dataset in datasets:
        try:
            test_embedding_data(dataset)
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"Error testing {dataset}: {e}")
            print("\n" + "="*50 + "\n") 