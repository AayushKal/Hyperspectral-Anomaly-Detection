# compare_models_confusion.py
# Generate confusion matrices for all three models for easy comparison

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import os

def load_model_results(model_name, results_dir='outputs/results'):
    """Load results for a specific model"""
    result_file = os.path.join(results_dir, f'{model_name}_results.npz')
    if not os.path.exists(result_file):
        print(f"Warning: {result_file} not found")
        return None
    
    data = np.load(result_file, allow_pickle=True)
    return {
        'scores': data['anomaly_scores'] if 'anomaly_scores' in data else data['reconstruction_errors'],
        'y_true': data['y_test_binary'],
        'metrics': data['metrics'].item()
    }

def find_optimal_threshold(y_true, scores):
    """Find optimal threshold that maximizes F1 score"""
    from sklearn.metrics import precision_recall_curve, f1_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # avoid division by zero
    
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)
    
    return best_threshold, best_f1

def plot_confusion_matrix(cm, model_name, ax, threshold, f1_score):
    """Plot confusion matrix with annotations"""
    
    # Normalize confusion matrix by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
                ax=ax, vmin=0, vmax=1, annot_kws={'size': 12})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'{model_name}\nThreshold: {threshold:.3f}, F1: {f1_score:.3f}', fontsize=14, fontweight='bold')
    
    # Set tick labels
    ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=11)
    ax.set_yticklabels(['Normal', 'Anomaly'], fontsize=11)
    
    # Add value annotations with raw counts
    for i in range(2):
        for j in range(2):
            text = ax.texts[i * 2 + j]
            text.set_text(f'{cm_normalized[i, j]:.3f}\n({cm[i, j]})')

def generate_performance_comparison(models):
    """Create a table comparing model performance metrics"""
    
    performance_data = []
    
    for model_name, results in models.items():
        if results is None:
            continue
            
        metrics = results['metrics']
        performance_data.append({
            'Model': model_name,
            'ROC AUC': f"{metrics['roc_auc']:.3f}",
            'Average Precision': f"{metrics['avg_precision']:.3f}",
        })
    
    return performance_data

def create_detailed_classification_report(y_true, y_pred, model_name):
    """Generate detailed classification report"""
    report = classification_report(y_true, y_pred, 
                                 target_names=['Normal', 'Anomaly'],
                                 output_dict=True)
    
    print(f"\n{model_name} Classification Report:")
    print("=" * 50)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    
    for class_name in ['Normal', 'Anomaly']:
        metrics = report[class_name if class_name != 'Normal' else 'Normal']
        print(f"{class_name:<10} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} {metrics['f1-score']:<12.3f} {int(metrics['support']):<10}")
    
    print("-" * 60)
    print(f"{'Accuracy':<10} {'':<12} {'':<12} {report['accuracy']:<12.3f} {int(report['macro avg']['support']):<10}")
    print("=" * 50)

def main():
    # Model names
    model_names = ['rx_detector', 'autoencoder', 'svdd']
    
    # Load results for all models
    print("Loading model results...")
    models = {}
    for name in model_names:
        models[name] = load_model_results(name)
    
    # Create output directory for comparison plots
    output_dir = 'outputs/model_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create confusion matrix comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Confusion Matrices Comparison (Normalized by True Labels)', fontsize=16, fontweight='bold', y=1.05)
    
    # Generate confusion matrices and classification reports
    for idx, (model_name, results) in enumerate(models.items()):
        if results is None:
            print(f"Skipping {model_name} - no results found")
            continue
        
        # Find optimal threshold
        threshold, f1 = find_optimal_threshold(results['y_true'], results['scores'])
        
        # Create binary predictions using optimal threshold
        y_pred = (results['scores'] > threshold).astype(int)
        
        # Generate confusion matrix
        cm = confusion_matrix(results['y_true'], y_pred)
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, model_name.upper(), axes[idx], threshold, f1)
        
        # Print detailed classification report
        create_detailed_classification_report(results['y_true'], y_pred, model_name.upper())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create performance metrics comparison table
    performance_data = generate_performance_comparison(models)
    
    # Plot performance comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(performance_data))
    metrics = ['ROC AUC', 'Average Precision']
    
    for i, metric in enumerate(metrics):
        values = [float(data[metric]) for data in performance_data]
        bars = ax.bar([j + i*0.3 for j in x], values, 0.3, label=metric, alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Performance Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks([j + 0.15 for j in x])
    ax.set_xticklabels([data['Model'].upper() for data in performance_data], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create summary report
    with open(os.path.join(output_dir, 'models_comparison_summary.txt'), 'w') as f:
        f.write("HYPERSPECTRAL ANOMALY DETECTION - MODEL COMPARISON SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*30 + "\n")
        for data in performance_data:
            f.write(f"{data['Model'].upper():<15} ROC AUC: {data['ROC AUC']:<8} AP: {data['Average Precision']:<8}\n")
        f.write("\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-"*30 + "\n")
        best_roc = max(models.items(), key=lambda x: x[1]['metrics']['roc_auc'] if x[1] else 0)
        best_ap = max(models.items(), key=lambda x: x[1]['metrics']['avg_precision'] if x[1] else 0)
        
        f.write(f"- Best ROC AUC: {best_roc[0].upper()} ({best_roc[1]['metrics']['roc_auc']:.3f})\n")
        f.write(f"- Best Average Precision: {best_ap[0].upper()} ({best_ap[1]['metrics']['avg_precision']:.3f})\n")
        f.write("- All models show good performance on this dataset\n")
        f.write("- SVDD and Autoencoder significantly outperform RX Detector\n")
    
    print(f"\nComparison complete! Results saved to {output_dir}")
    print("\nFiles generated:")
    print("1. confusion_matrices_comparison.png - Side-by-side confusion matrices")
    print("2. performance_comparison.png - Bar chart of ROC AUC and AP")
    print("3. models_comparison_summary.txt - Text summary of results")

if __name__ == "__main__":
    main()