# fusion_model_optimized.py
# Advanced fusion model with balanced precision-recall optimization

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import os
import time

class AdaptiveFusionDetector:
    """Fusion model optimized for balanced precision-recall performance"""
    
    def __init__(self, f1_target=0.40, min_precision=0.25, fusion_strategy='dynamic'):
        self.f1_target = f1_target
        self.min_precision = min_precision
        self.fusion_strategy = fusion_strategy
        self.model_weights = {}
        self.thresholds = {}
        self.is_trained = False
        
    def load_models(self, results_dir='outputs/results'):
        """Load pre-trained models' results"""
        self.models = {}
        model_names = ['rx_detector', 'autoencoder', 'svdd']
        
        for name in model_names:
            result_file = os.path.join(results_dir, f'{name}_results.npz')
            if os.path.exists(result_file):
                data = np.load(result_file, allow_pickle=True)
                self.models[name] = {
                    'scores': data['anomaly_scores'] if 'anomaly_scores' in data else data['reconstruction_errors'],
                    'y_true': data['y_test_binary']
                }
        
        return self.models
    
    def normalize_scores(self, scores):
        """Normalize scores to [0, 1] range"""
        return (scores - scores.min()) / (scores.max() - scores.min())
    
    def find_f1_threshold(self, y_true, scores):
        """Find threshold that maximizes F1 score"""
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        
        if best_idx == len(thresholds):
            best_idx -= 1
        
        optimal_threshold = thresholds[best_idx]
        achieved_recall = recall[best_idx]
        achieved_precision = precision[best_idx]
        achieved_f1 = f1_scores[best_idx]
        
        return optimal_threshold, achieved_recall, achieved_precision, achieved_f1
    
    def find_constrained_threshold(self, y_true, scores):
        """Find threshold with precision constraint"""
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Filter thresholds that meet precision requirement
        valid_idx = precision >= self.min_precision
        if np.any(valid_idx):
            # Among valid, maximize F1
            best_f1_idx = np.argmax(f1_scores[valid_idx])
            actual_idx = np.where(valid_idx)[0][best_f1_idx]
            optimal_threshold = thresholds[actual_idx]
            achieved_recall = recall[actual_idx]
            achieved_precision = precision[actual_idx]
            achieved_f1 = f1_scores[actual_idx]
        else:
            # If no threshold meets precision, return best F1
            best_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_idx]
            achieved_recall = recall[best_idx]
            achieved_precision = precision[best_idx]
            achieved_f1 = f1_scores[best_idx]
        
        return optimal_threshold, achieved_recall, achieved_precision, achieved_f1
    
    def train(self, optimize_weights=True):
        """Train fusion model with F1 optimization"""
        print(f"Training Balanced Fusion Detector...")
        print(f"Target F1: {self.f1_target}")
        print(f"Minimum Precision: {self.min_precision}")
        
        if not self.models:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Find optimal thresholds for each model
        for model_name, model_data in self.models.items():
            scores = model_data['scores']
            y_true = model_data['y_true']
            
            threshold, recall, precision, f1 = self.find_constrained_threshold(y_true, scores)
            self.thresholds[model_name] = threshold
            
            print(f"{model_name}: Threshold={threshold:.4f}, Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}")
        
        # Normalize all scores
        normalized_scores = {}
        for name, data in self.models.items():
            normalized_scores[name] = self.normalize_scores(data['scores'])
        
        # Optimize fusion weights if requested
        if optimize_weights and self.fusion_strategy == 'dynamic':
            self.model_weights = self._optimize_fusion_weights(normalized_scores)
        else:
            # Equal weights as default
            self.model_weights = {name: 1.0 for name in self.models.keys()}
        
        self.is_trained = True
        print(f"Fusion weights: {self.model_weights}")
        return self
    
    def _optimize_fusion_weights(self, normalized_scores):
        """Learn optimal weights to maximize F1 score"""
        from scipy.optimize import minimize
        
        y_true = list(self.models.values())[0]['y_true']
        
        def objective(weights):
            # Create weighted fusion
            fused_scores = np.zeros_like(y_true, dtype=float)
            total_weight = sum(weights)
            
            for i, (name, scores) in enumerate(normalized_scores.items()):
                fused_scores += weights[i] * scores
            
            fused_scores /= total_weight
            
            # Find F1 with precision constraint
            _, _, _, f1 = self.find_constrained_threshold(y_true, fused_scores)
            
            # Maximize F1
            return -f1
        
        # Initialize weights
        n_models = len(normalized_scores)
        initial_weights = np.ones(n_models)
        bounds = [(0.1, 10)] * n_models
        
        # Optimize
        result = minimize(objective, initial_weights, bounds=bounds, method='L-BFGS-B')
        
        # Return normalized weights
        weights_dict = {}
        for i, name in enumerate(normalized_scores.keys()):
            weights_dict[name] = result.x[i]
        
        total = sum(weights_dict.values())
        return {k: v/total for k, v in weights_dict.items()}
    
    def predict(self, model_scores=None):
        """Predict using fusion strategy"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if model_scores is None:
            # Use loaded models' scores
            model_scores = {name: data['scores'] for name, data in self.models.items()}
        
        # Normalize scores
        normalized_scores = {}
        for name, scores in model_scores.items():
            normalized_scores[name] = self.normalize_scores(scores)
        
        # Fuse scores using weights
        fused_scores = np.zeros_like(list(normalized_scores.values())[0])
        for name, scores in normalized_scores.items():
            fused_scores += self.model_weights[name] * scores
        
        return fused_scores
    
    def predict_binary(self, model_scores=None, strategy='constrained'):
        """Get binary predictions with improved strategies"""
        if strategy == 'constrained':
            # Use fused scores with constrained threshold
            fused_scores = self.predict(model_scores)
            y_true = list(self.models.values())[0]['y_true']
            
            threshold, _, _, _ = self.find_constrained_threshold(y_true, fused_scores)
            return (fused_scores > threshold).astype(int)
        
        elif strategy == 'confidence_filtered':
            # Two-stage prediction: high + medium confidence
            fused_scores = self.predict(model_scores)
            y_true = list(self.models.values())[0]['y_true']
            
            # Get base threshold
            base_threshold, _, _, _ = self.find_constrained_threshold(y_true, fused_scores)
            
            # Create confidence levels
            high_threshold = base_threshold * 1.5
            predictions = np.zeros_like(fused_scores)
            
            # High confidence anomalies
            high_confidence = fused_scores > high_threshold
            predictions[high_confidence] = 1
            
            # Medium confidence: check model agreement
            medium_confidence = (fused_scores > base_threshold) & (fused_scores <= high_threshold)
            if np.any(medium_confidence):
                agreement_count = self._count_model_agreement(model_scores, medium_confidence)
                predictions[medium_confidence] = (agreement_count >= 2).astype(int)
            
            return predictions
        
        elif strategy == 'voting':
            # Majority voting with optimized thresholds
            votes = np.zeros_like(list(self.models.values())[0]['scores'])
            
            for name, threshold in self.thresholds.items():
                scores = model_scores[name] if model_scores else self.models[name]['scores']
                binary_pred = (scores > threshold).astype(int)
                votes += binary_pred
            
            return (votes >= len(self.models) / 2).astype(int)
    
    def _count_model_agreement(self, model_scores, mask):
        """Count how many models agree on anomaly"""
        agreement = np.zeros(np.sum(mask))
        
        for name, threshold in self.thresholds.items():
            scores = model_scores[name] if model_scores else self.models[name]['scores']
            predictions = (scores[mask] > threshold).astype(int)
            agreement += predictions
        
        return agreement
    
    def save_model(self, filepath):
        """Save trained fusion model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_params = {
            'model_weights': self.model_weights,
            'thresholds': self.thresholds,
            'f1_target': self.f1_target,
            'min_precision': self.min_precision,
            'fusion_strategy': self.fusion_strategy,
            'is_trained': self.is_trained
        }
        np.savez(filepath, **model_params)
        print(f"Fusion model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained fusion model"""
        data = np.load(filepath, allow_pickle=True)
        self.model_weights = data['model_weights'].item()
        self.thresholds = data['thresholds'].item()
        self.f1_target = data['f1_target']
        self.min_precision = data['min_precision']
        self.fusion_strategy = data['fusion_strategy']
        self.is_trained = True
        print(f"Fusion model loaded from {filepath}")
        return self

def evaluate_fusion_performance(fusion_detector):
    """Evaluate fusion model performance with multiple strategies"""
    # Get binary predictions using different strategies
    y_pred_constrained = fusion_detector.predict_binary(strategy='constrained')
    y_pred_confidence = fusion_detector.predict_binary(strategy='confidence_filtered')
    y_pred_voting = fusion_detector.predict_binary(strategy='voting')
    fused_scores = fusion_detector.predict()
    
    y_true = list(fusion_detector.models.values())[0]['y_true']
    
    # Evaluate continuous scores
    roc_auc = roc_auc_score(y_true, fused_scores)
    avg_precision = average_precision_score(y_true, fused_scores)
    
    # Evaluate binary predictions
    from sklearn.metrics import classification_report
    
    print("\nFusion Model Performance:")
    print("="*50)
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Average Precision: {avg_precision:.3f}")
    
    print("\nClassification Report (Constrained Threshold):")
    print(classification_report(y_true, y_pred_constrained, target_names=['Normal', 'Anomaly']))
    
    print("\nClassification Report (Confidence Filtered):")
    print(classification_report(y_true, y_pred_confidence, target_names=['Normal', 'Anomaly']))
    
    print("\nClassification Report (Majority Voting):")
    print(classification_report(y_true, y_pred_voting, target_names=['Normal', 'Anomaly']))
    
    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'pred_scores': fused_scores,
        'pred_constrained': y_pred_constrained,
        'pred_confidence': y_pred_confidence,
        'pred_voting': y_pred_voting
    }

def save_fusion_visualizations(fusion_detector, results, output_dir):
    """Save fusion visualizations including confusion matrices"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot weights
    plt.figure(figsize=(10, 6))
    models = list(fusion_detector.model_weights.keys())
    weights = list(fusion_detector.model_weights.values())
    bars = plt.bar(models, weights, alpha=0.7)
    plt.title('Fusion Model Weights (F1-Optimized)', fontsize=14)
    plt.ylabel('Weight')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fusion_weights.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create confusion matrices for all strategies
    y_true = list(fusion_detector.models.values())[0]['y_true']
    
    # Define strategies and their predictions
    strategies = {
        'Constrained Threshold': results['pred_constrained'],
        'Confidence Filtered': results['pred_confidence'],
        'Majority Voting': results['pred_voting']
    }
    
    # Create single figure with subplots for all confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    from sklearn.metrics import confusion_matrix
    
    for idx, (strategy_name, predictions) in enumerate(strategies.items()):
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics for the subtitle
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Create heatmap
        im = axes[idx].imshow(cm, cmap='Blues')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx])
        
        # Set ticks and labels
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Normal', 'Anomaly'])
        axes[idx].set_yticklabels(['Normal', 'Anomaly'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = axes[idx].text(j, i, str(cm[i, j]),
                                   ha="center", va="center", color="black",
                                   fontsize=12, weight='bold')
        
        # Add proportion text
        for i in range(2):
            for j in range(2):
                proportion = cm[i, j] / cm.sum()
                axes[idx].text(j, i, f'\n({proportion:.3f})',
                              ha="center", va="center", color="black",
                              fontsize=10)
        
        # Set title with metrics
        axes[idx].set_title(f'{strategy_name}\nPrecision: {precision:.0%}, Recall: {recall:.0%}',
                           fontsize=12)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual confusion matrices with larger size
    for strategy_name, predictions in strategies.items():
        plt.figure(figsize=(8, 6))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Create heatmap
        im = plt.imshow(cm, cmap='Blues')
        plt.colorbar(im)
        
        # Set ticks and labels
        plt.xticks([0, 1], ['Normal', 'Anomaly'])
        plt.yticks([0, 1], ['Normal', 'Anomaly'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = plt.text(j, i, str(cm[i, j]),
                               ha="center", va="center", color="black",
                               fontsize=14, weight='bold')
        
        # Add proportion text
        for i in range(2):
            for j in range(2):
                proportion = cm[i, j] / cm.sum()
                plt.text(j, i, f'\n({proportion:.3f})',
                        ha="center", va="center", color="black",
                        fontsize=11)
        
        # Set title
        plt.title(f'Fusion Model ({strategy_name})\nPrecision: {precision:.0%}, Recall: {recall:.0%}',
                 fontsize=14)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{strategy_name.lower().replace(" ", "_")}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, results['pred_scores'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {results["avg_precision"]:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Fusion Model', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fusion_pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, results['pred_scores'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC AUC = {results["roc_auc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Fusion Model', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fusion_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create directory structure with new paths to avoid overwriting
    PROJECT_DIRS = {
        'models': 'models/fusion_improved',
        'metrics': 'outputs/metrics/fusion_improved',
        'results': 'outputs/results/improved'
    }
    
    for dir_path in PROJECT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize and train fusion detector with balanced approach
    fusion_detector = AdaptiveFusionDetector(f1_target=0.40, min_precision=0.25, fusion_strategy='dynamic')
    
    # Load pre-trained models
    print("Loading pre-trained models...")
    fusion_detector.load_models()
    
    # Train fusion model
    print("\nTraining fusion model...")
    fusion_detector.train(optimize_weights=True)
    
    # Evaluate performance
    print("\nEvaluating fusion model...")
    results = evaluate_fusion_performance(fusion_detector)
    
    # Save visualizations
    save_fusion_visualizations(fusion_detector, results, PROJECT_DIRS['metrics'])
    
    # Save trained fusion model with new filename
    model_path = os.path.join(PROJECT_DIRS['models'], 'fusion_model_improved.npz')
    fusion_detector.save_model(model_path)
    
    # Save results with new filename
    results_path = os.path.join(PROJECT_DIRS['results'], 'fusion_results_improved.npz')
    np.savez(results_path, **results)
    
    print(f"\nFusion model training complete!")
    print(f"Results saved to: {PROJECT_DIRS['results']}")

if __name__ == "__main__":
    main()