'''
Adversarial Hierarchical-Aware Edge Attention Learning Method for Network Intrusion Detection
@mrforesthao
'''
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(results, G, label_encoder):
    # 提取真实标签
    coarse_true = G.edata['coarse_label'].cpu().numpy()
    fine_true = G.edata['fine_label'].cpu().numpy()
    
    # 粗粒度报告
    coarse_pred = results['is_attack'].numpy()
    print("Coarse-Level Performance:")
    print(classification_report(coarse_true, coarse_pred, target_names=['Normal', 'Attack']))
    
    # 细粒度报告（仅对真实攻击样本）
    attack_mask = (coarse_true == 1)
    fine_pred = results['attack_types'].numpy()
    valid_mask = (fine_pred != -1) & attack_mask
    
    if valid_mask.sum() > 0:
        print("\nFine-Level Performance:")
        print(classification_report(
            fine_true[valid_mask], 
            fine_pred[valid_mask],
            target_names=label_encoder.classes_,
            zero_division=0
        ))
        
        # 混淆矩阵可视化
        cm = confusion_matrix(fine_true[valid_mask], fine_pred[valid_mask])
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title('Attack Type Confusion Matrix')
        plt.show()

def visualize_perturbation(original_feat, adv_feat):
    plt.figure(figsize=(10,4))
    
    plt.subplot(1,2,1)
    plt.hist(original_feat.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Original')
    plt.hist(adv_feat.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Adversarial')
    plt.title('Feature Distribution')
    plt.legend()
    
    plt.subplot(1,2,2)
    delta = (adv_feat - original_feat).abs().mean(dim=(1,2)).cpu().numpy()
    plt.plot(delta)
    plt.title('Perturbation Magnitude per Node')
    
    plt.tight_layout()
    plt.show()