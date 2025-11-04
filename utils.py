'''
Adversarial Hierarchical-Aware Edge Attention Learning Method for Network Intrusion Detection
@mrforesthao
'''
import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools

def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('confusion_matrix.pdf', format='pdf')
    plt.show()

def evaluate_robustness(model, G, adversary, device):
    model.eval()
    
    # 干净样本准确率
    clean_acc = compute_accuracy(model(G), G.edata['Label'])
    
    # 对抗样本准确率
    adv_G = adversary.generate(G)
    adv_acc = compute_accuracy(model(adv_G), G.edata['Label'])
    
    # 鲁棒性得分
    robustness = adv_acc / clean_acc
    return {
        'clean_acc': clean_acc,
        'adv_acc': adv_acc,
        'robustness': robustness
    }

def evaluate_robustness_PGD(model, test_loader, adversary, device):
    model.eval()
    total = 0
    correct_clean = 0
    correct_adv = 0
    confidence_shifts = []
    
    for g, labels in test_loader:
        g = g.to(device)
        labels = labels.to(device)
        
        # 原始样本准确率
        with torch.no_grad():
            outputs = model(g)
            preds_clean = outputs.argmax(1)
            correct_clean += (preds_clean == labels).sum().item()
        
        # 生成对抗样本
        adv_feats = adversary.generate(g, labels)
        g.edata['h'] = adv_feats
        
        # 对抗样本准确率
        with torch.no_grad():
            outputs_adv = model(g)
            preds_adv = outputs_adv.argmax(1)
            correct_adv += (preds_adv == labels).sum().item()
            
            # 置信度偏移计算
            prob_clean = F.softmax(outputs, dim=1)
            prob_adv = F.softmax(outputs_adv, dim=1)
            shift = (prob_adv - prob_clean).abs().mean().item()
            confidence_shifts.append(shift)
        
        total += labels.size(0)
    
    clean_acc = correct_clean / total
    robust_acc = correct_adv / total
    asr = 1 - robust_acc
    mean_shift = sum(confidence_shifts) / len(confidence_shifts)
    
    return {
        'clean_accuracy': clean_acc,
        'robust_accuracy': robust_acc,
        'asr': asr,
        'mean_confidence_shift': mean_shift
    }