# fusion_confusion_matrix.py
# Generate confusion matrix for fusion model

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_fusion_confusion_matrices():
    """Generate confusion matrices for fusion model predictions"""
    
    # Load fusion results
    results = np.load('outputs/results/fusion_results.npz', allow_pickle=True)
    
    # Get true labels
    test_data = np.load('outputs/results/rx_detector_results.npz', allow_pickle=True)
    y_true = test_data['y_test_binary']
    
    # Get predictions
    y_pred_binary = results['pred_binary']
    y_pred_voting = results['pred_voting']
    
    # Create figure with two confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Recall-Optimized confusion matrix
    cm_recall = confusion_matrix(y_true, y_pred_binary)
    cm_recall_norm = cm_recall.astype('float') / cm_recall.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_recall_norm, annot=True, fmt='.3f', cmap='Blues', ax=axes[0])
    axes[0].set_title('Fusion Model (Recall-Optimized)\nPrecision: 19%, Recall: 85%', fontsize=12)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_xticklabels(['Normal', 'Anomaly'])
    axes[0].set_yticklabels(['Normal', 'Anomaly'])
    
    # Add absolute values as text
    for i in range(2):
        for j in range(2):
            text = axes[0].texts[i * 2 + j]
            text.set_text(f'{cm_recall_norm[i, j]:.3f}\n({cm_recall[i, j]})')
    
    # Plot Majority Voting confusion matrix
    cm_voting = confusion_matrix(y_true, y_pred_voting)
    cm_voting_norm = cm_voting.astype('float') / cm_voting.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_voting_norm, annot=True, fmt='.3f', cmap='Blues', ax=axes[1])
    axes[1].set_title('Fusion Model (Majority Voting)\nPrecision: 16%, Recall: 90%', fontsize=12)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_xticklabels(['Normal', 'Anomaly'])
    axes[1].set_yticklabels(['Normal', 'Anomaly'])
    
    # Add absolute values as text
    for i in range(2):
        for j in range(2):
            text = axes[1].texts[i * 2 + j]
            text.set_text(f'{cm_voting_norm[i, j]:.3f}\n({cm_voting[i, j]})')
    
    plt.tight_layout()
    plt.savefig('outputs/metrics/fusion/fusion_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create single large confusion matrix with all metrics
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate metrics for Recall-Optimized
    tn, fp, fn, tp = cm_recall.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Plot confusion matrix with detailed annotations
    cm_display = cm_recall_norm
    sns.heatmap(cm_display, annot=False, fmt='.3f', cmap='Blues', ax=ax)
    
    # Custom annotations with detailed metrics
    ax.text(0.5, 0.5, f'TN: {tn}\n{cm_display[0,0]:.3f}', ha='center', va='center', fontsize=12, color='black' if cm_display[0,0] < 0.5 else 'white')
    ax.text(1.5, 0.5, f'FP: {fp}\n{cm_display[0,1]:.3f}\nFPR: {fpr:.3f}', ha='center', va='center', fontsize=12, color='black' if cm_display[0,1] < 0.5 else 'white')
    ax.text(0.5, 1.5, f'FN: {fn}\n{cm_display[1,0]:.3f}', ha='center', va='center', fontsize=12, color='black' if cm_display[1,0] < 0.5 else 'white')
    ax.text(1.5, 1.5, f'TP: {tp}\n{cm_display[1,1]:.3f}', ha='center', va='center', fontsize=12, color='black' if cm_display[1,1] < 0.5 else 'white')
    
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('True', fontsize=14)
    ax.set_title(f'Fusion Model Confusion Matrix (Recall-Optimized)\nPrecision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Accuracy: {accuracy:.3f}', fontsize=14)
    ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=12)
    ax.set_yticklabels(['Normal', 'Anomaly'], fontsize=12)
    
    plt.tight_layout()
    plt.savefig('outputs/metrics/fusion/fusion_detailed_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create performance comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Precision', 'Recall', 'F1-Score', 'FPR']
    recall_opt_values = [precision, recall, f1, fpr]
    
    # Calculate metrics for Voting
    tn_v, fp_v, fn_v, tp_v = cm_voting.ravel()
    precision_v = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 0
    recall_v = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0
    f1_v = 2 * (precision_v * recall_v) / (precision_v + recall_v) if (precision_v + recall_v) > 0 else 0
    fpr_v = fp_v / (fp_v + tn_v) if (fp_v + tn_v) > 0 else 0
    
    voting_values = [precision_v, recall_v, f1_v, fpr_v]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, recall_opt_values, width, label='Recall-Optimized', alpha=0.8)
    bars2 = ax.bar(x + width/2, voting_values, width, label='Majority Voting', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Fusion Model Performance Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig('outputs/metrics/fusion/fusion_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Fusion confusion matrices generated!")
    print("Files saved:")
    print("1. outputs/metrics/fusion/fusion_confusion_matrices.png")
    print("2. outputs/metrics/fusion/fusion_detailed_confusion_matrix.png")
    print("3. outputs/metrics/fusion/fusion_metrics_comparison.png")

if __name__ == "__main__":
    plot_fusion_confusion_matrices()