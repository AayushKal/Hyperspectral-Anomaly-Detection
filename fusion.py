# fusion_model_organized.py
# Advanced fusion model optimized for high anomaly detection

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import os
import time

class AdaptiveFusionDetector:
    """Fusion model that adapts to prioritize recall"""
    
    def __init__(self, recall_target=0.80, fusion_strategy='dynamic'):
        self.recall_target = recall_target
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
    
    def find_recall_threshold(self, y_true, scores, target_recall):
        """Find threshold that achieves target recall"""
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        
        recall_diff = np.abs(recall - target_recall)
        best_idx = np.argmin(recall_diff)
        
        if best_idx == len(thresholds):
            best_idx -= 1
        
        optimal_threshold = thresholds[best_idx]
        achieved_recall = recall[best_idx]
        achieved_precision = precision[best_idx]
        
        return optimal_threshold, achieved_recall, achieved_precision
    
    def train(self, optimize_weights=True):
        """Train fusion model with recall optimization"""
        print(f"Training Adaptive Fusion Detector...")
        print(f"Target Recall: {self.recall_target}")
        
        if not self.models:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        for model_name, model_data in self.models.items():
            scores = model_data['scores']
            y_true = model_data['y_true']
            
            threshold, recall, precision = self.find_recall_threshold(y_true, scores, self.recall_target)
            self.thresholds[model_name] = threshold
            
            print(f"{model_name}: Threshold={threshold:.4f}, Recall={recall:.3f}, Precision={precision:.3f}")
        
        normalized_scores = {}
        for name, data in self.models.items():
            normalized_scores[name] = self.normalize_scores(data['scores'])
        
        if optimize_weights and self.fusion_strategy == 'dynamic':
            self.model_weights = self._optimize_fusion_weights(normalized_scores)
        else:
            self.model_weights = {name: 1.0 for name in self.models.keys()}
        
        self.is_trained = True
        print(f"Fusion weights: {self.model_weights}")
        return self
    
    def _optimize_fusion_weights(self, normalized_scores):
        """Learn optimal weights to maximize recall"""
        from scipy.optimize import minimize
        
        y_true = list(self.models.values())[0]['y_true']
        
        def objective(weights):
            fused_scores = np.zeros_like(y_true, dtype=float)
            total_weight = sum(weights)
            
            for i, (name, scores) in enumerate(normalized_scores.items()):
                fused_scores += weights[i] * scores
            
            fused_scores /= total_weight
            
            _, recall, _ = self.find_recall_threshold(y_true, fused_scores, self.recall_target)
            
            return -recall
        
        n_models = len(normalized_scores)
        initial_weights = np.ones(n_models)
        bounds = [(0.1, 10)] * n_models
        
        result = minimize(objective, initial_weights, bounds=bounds, method='L-BFGS-B')
        
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
            model_scores = {name: data['scores'] for name, data in self.models.items()}
        
        normalized_scores = {}
        for name, scores in model_scores.items():
            normalized_scores[name] = self.normalize_scores(scores)
        
        fused_scores = np.zeros_like(list(normalized_scores.values())[0])
        for name, scores in normalized_scores.items():
            fused_scores += self.model_weights[name] * scores
        
        return fused_scores
    
    def predict_binary(self, model_scores=None, use_individual_thresholds=False):
        """Get binary predictions"""
        if use_individual_thresholds:
            votes = np.zeros_like(list(self.models.values())[0]['scores'])
            
            for name, threshold in self.thresholds.items():
                scores = model_scores[name] if model_scores else self.models[name]['scores']
                binary_pred = (scores > threshold).astype(int)
                votes += binary_pred
            
            return (votes >= len(self.models) / 2).astype(int)
        else:
            fused_scores = self.predict(model_scores)
            y_true = list(self.models.values())[0]['y_true']
            
            threshold, _, _ = self.find_recall_threshold(y_true, fused_scores, self.recall_target)
            return (fused_scores > threshold).astype(int)
    
    def save_model(self, filepath):
        """Save trained fusion model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_params = {
            'model_weights': self.model_weights,
            'thresholds': self.thresholds,
            'recall_target': self.recall_target,
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
        self.recall_target = data['recall_target']
        self.fusion_strategy = data['fusion_strategy']
        self.is_trained = True
        print(f"Fusion model loaded from {filepath}")
        return self

def evaluate_fusion_performance(fusion_detector):
    """Evaluate fusion model performance"""
    y_pred_binary = fusion_detector.predict_binary()
    y_pred_voting = fusion_detector.predict_binary(use_individual_thresholds=True)
    fused_scores = fusion_detector.predict()
    
    y_true = list(fusion_detector.models.values())[0]['y_true']
    
    roc_auc = roc_auc_score(y_true, fused_scores)
    avg_precision = average_precision_score(y_true, fused_scores)
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\nFusion Model Performance:")
    print("="*50)
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Average Precision: {avg_precision:.3f}")
    
    print("\nClassification Report (Recall-Optimized Threshold):")
    print(classification_report(y_true, y_pred_binary, target_names=['Normal', 'Anomaly']))
    
    print("\nClassification Report (Majority Voting):")
    print(classification_report(y_true, y_pred_voting, target_names=['Normal', 'Anomaly']))
    
    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'pred_scores': fused_scores,
        'pred_binary': y_pred_binary,
        'pred_voting': y_pred_voting
    }

def save_fusion_visualizations(fusion_detector, results, output_dir):
    """Save fusion visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    models = list(fusion_detector.model_weights.keys())
    weights = list(fusion_detector.model_weights.values())
    bars = plt.bar(models, weights, alpha=0.7)
    plt.title('Fusion Model Weights (Recall-Optimized)', fontsize=14)
    plt.ylabel('Weight')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fusion_weights.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    y_true = list(fusion_detector.models.values())[0]['y_true']
    fpr, tpr, _ = roc_curve(y_true, results['pred_scores'])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC AUC = {results["roc_auc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve - Fusion Model', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fusion_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    PROJECT_DIRS = {
        'models': 'models/fusion',
        'metrics': 'outputs/metrics/fusion',
        'results': 'outputs/results'
    }
    
    for dir_path in PROJECT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)
    
    fusion_detector = AdaptiveFusionDetector(recall_target=0.85, fusion_strategy='dynamic')
    
    print("Loading pre-trained models...")
    fusion_detector.load_models()
    
    print("\nTraining fusion model...")
    fusion_detector.train(optimize_weights=True)
    
    print("\nEvaluating fusion model...")
    results = evaluate_fusion_performance(fusion_detector)
    
    save_fusion_visualizations(fusion_detector, results, PROJECT_DIRS['metrics'])
    
    model_path = os.path.join(PROJECT_DIRS['models'], 'fusion_model.npz')
    fusion_detector.save_model(model_path)
    
    results_path = os.path.join(PROJECT_DIRS['results'], 'fusion_results.npz')
    np.savez(results_path, **results)
    
    print(f"\nFusion model training complete!")
    print(f"Results saved to: {PROJECT_DIRS['results']}")

if __name__ == "__main__":
    main()