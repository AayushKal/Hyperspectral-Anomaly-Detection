# svdd_organized.py
# Support Vector Data Description (SVDD) anomaly detection implementation for hyperspectral images

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV
import joblib
import os
import time
from sklearn.base import clone

class SVDDDetector:
    """Support Vector Data Description for anomaly detection"""
    
    def __init__(self, kernel='rbf', nu=0.1, gamma='scale'):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.model = None
        self.is_trained = False
        self.best_params = None
        
    def tune_parameters(self, X_train, cv=5):
        """Find optimal hyperparameters using grid search"""
        print("Tuning SVDD hyperparameters...")
        
        param_grid = {
            'nu': [0.01, 0.05, 0.1, 0.2],
            'gamma': [0.001, 0.01, 0.1, 1.0]
        }
        
        base_model = OneClassSVM(kernel=self.kernel)
        
        if len(X_train) > 5000:
            print(f"Using subset of {5000} samples for parameter tuning...")
            tune_indices = np.random.choice(len(X_train), size=5000, replace=False)
            X_tune = X_train[tune_indices]
        else:
            X_tune = X_train
        
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=cv,
            n_jobs=-1,
            verbose=1,
            scoring=lambda estimator, X: -np.mean(np.abs(estimator.decision_function(X)))
        )
        
        grid_search.fit(X_tune)
        
        self.best_params = grid_search.best_params_
        self.nu = self.best_params['nu']
        self.gamma = self.best_params['gamma']
        
        print(f"Best parameters found: nu={self.nu}, gamma={self.gamma}")
        return self.best_params
    
    def fit(self, X, tune_params=True):
        """Train the SVDD on normal data"""
        print(f"Training SVDD on data shape: {X.shape}")
        print(f"Parameters: kernel={self.kernel}, nu={self.nu}, gamma={self.gamma}")
        
        if tune_params:
            self.tune_parameters(X)
        
        self.model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma
        )
        
        start_time = time.time()
        self.model.fit(X)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Calculate anomaly scores for data X"""
        if not self.is_trained:
            raise ValueError("Detector must be trained before prediction")
        
        decision_scores = self.model.decision_function(X)
        
        anomaly_scores = -decision_scores
        
        return anomaly_scores
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Detector must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_params = {
            'model': self.model,
            'kernel': self.kernel,
            'nu': self.nu,
            'gamma': self.gamma,
            'is_trained': self.is_trained,
            'best_params': self.best_params
        }
        joblib.dump(model_params, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        model_params = joblib.load(filepath)
        self.model = model_params['model']
        self.kernel = model_params['kernel']
        self.nu = model_params['nu']
        self.gamma = model_params['gamma']
        self.is_trained = model_params['is_trained']
        self.best_params = model_params.get('best_params', None)
        print(f"Model loaded from {filepath}")
        return self

def define_anomaly_labels(y, anomaly_classes):
    """Define which classes are anomalies"""
    y_binary = np.zeros_like(y)
    for anomaly_class in anomaly_classes:
        y_binary[y == anomaly_class] = 1
    return y_binary

def evaluate_detector(y_true, scores):
    """Evaluate anomaly detection performance"""
    roc_auc = roc_auc_score(y_true, scores)
    
    avg_precision = average_precision_score(y_true, scores)
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    
    precision, recall, _ = precision_recall_curve(y_true, scores)
    
    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'roc_curve': (fpr, tpr),
        'pr_curve': (precision, recall)
    }

def reconstruct_test_data(values, original_shape, non_background_idx, idx_test):
    """Reconstruct 2D image from test data values"""
    rows, cols, bands = original_shape
    full_array = np.zeros(rows * cols)
    
    non_bg_positions = np.where(non_background_idx)[0]
    
    test_positions = non_bg_positions[idx_test]
    
    full_array[test_positions] = values
    
    return full_array.reshape(rows, cols)

def save_metrics_plots(metrics, scores_2d, y_binary_2d, output_dir="outputs/metrics/svdd"):
    """Save all metric visualizations to separate output directory"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    fpr, tpr = metrics['roc_curve']
    plt.plot(fpr, tpr, label=f'ROC AUC = {metrics["roc_auc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - SVDD')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    precision, recall = metrics['pr_curve']
    plt.plot(recall, precision, label=f'AP = {metrics["avg_precision"]:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - SVDD')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(scores_2d, cmap='jet')
    plt.colorbar(label='Anomaly Score')
    plt.title('Anomaly Score Map')
    
    plt.subplot(1, 2, 2)
    plt.imshow(y_binary_2d, cmap='binary')
    plt.colorbar(label='Ground Truth (1=Anomaly)')
    plt.title('Ground Truth Anomalies')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/anomaly_maps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write("SVDD Performance Metrics\n")
        f.write("=======================\n\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.3f}\n")
        f.write(f"Average Precision: {metrics['avg_precision']:.3f}\n")
    
    print(f"Metrics and visualizations saved to: {output_dir}")

def main():
    PROJECT_DIRS = {
        'preprocessed_data': 'preprocessed_data',
        'models': 'models/svdd',
        'metrics': 'outputs/metrics/svdd',
        'results': 'outputs/results'
    }
    
    for dir_path in PROJECT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)
    
    print("Loading preprocessed data...")
    data = np.load(os.path.join(PROJECT_DIRS['preprocessed_data'], 'pavia_university_preprocessed.npz'))
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    idx_test = data['idx_test']
    original_shape = data['original_shape']
    non_background_idx = data['non_background_idx']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    unique_classes, counts = np.unique(y_train, return_counts=True)
    print("\nClass distribution in training data:")
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} samples")
    
    total_samples = len(y_train)
    minority_threshold = 0.05 * total_samples
    anomaly_classes = [cls for cls, count in zip(unique_classes, counts) if count < minority_threshold]
    print(f"\nAnomalies classes (minority): {anomaly_classes}")
    
    y_train_binary = define_anomaly_labels(y_train, anomaly_classes)
    y_test_binary = define_anomaly_labels(y_test, anomaly_classes)
    
    print(f"\nAnomalies in training set: {np.sum(y_train_binary)} out of {len(y_train_binary)} samples")
    print(f"Anomalies in test set: {np.sum(y_test_binary)} out of {len(y_test_binary)} samples")
    
    normal_mask = y_train_binary == 0
    X_train_normal = X_train[normal_mask]
    print(f"\nNormal training samples: {X_train_normal.shape[0]}")
    
    print("\nTraining SVDD detector...")
    svdd_detector = SVDDDetector()
    
    svdd_detector.fit(X_train_normal, tune_params=True)
    
    print("\nCalculating anomaly scores...")
    anomaly_scores = svdd_detector.predict(X_test)
    
    print("\nEvaluating performance...")
    metrics = evaluate_detector(y_test_binary, anomaly_scores)
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"Average Precision: {metrics['avg_precision']:.3f}")
    
    print("\nReconstructing anomaly score map...")
    scores_2d = reconstruct_test_data(anomaly_scores, original_shape, non_background_idx, idx_test)
    y_binary_2d = reconstruct_test_data(y_test_binary, original_shape, non_background_idx, idx_test)
    
    print("\nSaving metrics and visualizations...")
    save_metrics_plots(metrics, scores_2d, y_binary_2d, PROJECT_DIRS['metrics'])
    
    print("\nSaving trained model...")
    model_path = os.path.join(PROJECT_DIRS['models'], 'svdd_model.joblib')
    svdd_detector.save_model(model_path)
    
    results_path = os.path.join(PROJECT_DIRS['results'], 'svdd_results.npz')
    np.savez(results_path,
             anomaly_scores=anomaly_scores,
             anomaly_scores_2d=scores_2d,
             y_test_binary=y_test_binary,
             y_binary_2d=y_binary_2d,
             metrics=metrics)
    
    print("\nSVDD anomaly detection complete!")
    print(f"Results saved to: {PROJECT_DIRS['results']}")

if __name__ == "__main__":
    main()