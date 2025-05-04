import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import joblib
import os

class RXDetector:
    """Reed-Xiaoli (RX) Anomaly Detector for hyperspectral images"""
    
    def __init__(self):
        self.mean = None
        self.inv_cov = None
        self.is_trained = False
        
    def fit(self, X):
        """Train the RX detector on data X"""
        print(f"Training RX detector on data shape: {X.shape}")
        
        # Calculate mean
        self.mean = np.mean(X, axis=0)
        
        # Calculate covariance matrix
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered.T)
        
        # Calculate inverse covariance matrix (with regularization for stability)
        try:
            self.inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Add small regularization if matrix is singular
            print("Covariance matrix is singular, adding regularization...")
            reg_value = 1e-5
            self.inv_cov = np.linalg.inv(cov_matrix + reg_value * np.eye(cov_matrix.shape[0]))
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Calculate anomaly scores for data X"""
        if not self.is_trained:
            raise ValueError("Detector must be trained before prediction")
        
        # Center the data
        X_centered = X - self.mean
        
        # Calculate Mahalanobis distance for each sample
        scores = np.array([
            np.sqrt(np.dot(np.dot(x, self.inv_cov), x.T))
            for x in X_centered
        ])
        
        return scores
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Detector must be trained before saving")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_params = {
            'mean': self.mean,
            'inv_cov': self.inv_cov,
            'is_trained': self.is_trained
        }
        joblib.dump(model_params, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        model_params = joblib.load(filepath)
        self.mean = model_params['mean']
        self.inv_cov = model_params['inv_cov']
        self.is_trained = model_params['is_trained']
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
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_true, scores)
    
    # Calculate Average Precision
    avg_precision = average_precision_score(y_true, scores)
    
    # Get ROC curve points
    fpr, tpr, _ = roc_curve(y_true, scores)
    
    # Get Precision-Recall curve points
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
    
    # Get positions of all non-background pixels
    non_bg_positions = np.where(non_background_idx)[0]
    
    # Get positions of only test pixels
    test_positions = non_bg_positions[idx_test]
    
    # Place the values at test positions
    full_array[test_positions] = values
    
    return full_array.reshape(rows, cols)

def save_metrics_plots(metrics, scores_2d, y_binary_2d, output_dir="outputs/metrics/rx_detector"):
    """Save all metric visualizations to separate output directory"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr = metrics['roc_curve']
    plt.plot(fpr, tpr, label=f'ROC AUC = {metrics["roc_auc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - RX Detector')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot PR curve
    plt.figure(figsize=(8, 6))
    precision, recall = metrics['pr_curve']
    plt.plot(recall, precision, label=f'AP = {metrics["avg_precision"]:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - RX Detector')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot anomaly score heatmap
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
    
    # Create a summary metrics text file
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write("RX Detector Performance Metrics\n")
        f.write("===============================\n\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.3f}\n")
        f.write(f"Average Precision: {metrics['avg_precision']:.3f}\n")
    
    print(f"Metrics and visualizations saved to: {output_dir}")

def main():
    # Create organized directory structure
    PROJECT_DIRS = {
        'preprocessed_data': 'preprocessed_data',
        'models': 'models/rx_detector',
        'metrics': 'outputs/metrics/rx_detector',
        'results': 'outputs/results'
    }
    
    # Create all directories
    for dir_path in PROJECT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Load preprocessed data
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
    
    # Define anomaly classes
    unique_classes, counts = np.unique(y_train, return_counts=True)
    print("\nClass distribution in training data:")
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} samples")
    
    # Select classes with less than 5% of total samples as anomalies
    total_samples = len(y_train)
    minority_threshold = 0.05 * total_samples
    anomaly_classes = [cls for cls, count in zip(unique_classes, counts) if count < minority_threshold]
    print(f"\nAnomalies classes (minority): {anomaly_classes}")
    
    # Create binary labels for anomaly detection
    y_train_binary = define_anomaly_labels(y_train, anomaly_classes)
    y_test_binary = define_anomaly_labels(y_test, anomaly_classes)
    
    print(f"\nAnomalies in training set: {np.sum(y_train_binary)} out of {len(y_train_binary)} samples")
    print(f"Anomalies in test set: {np.sum(y_test_binary)} out of {len(y_test_binary)} samples")
    
    # Train RX detector
    print("\nTraining RX detector...")
    rx_detector = RXDetector()
    rx_detector.fit(X_train)
    
    # Predict on test data
    print("Calculating anomaly scores...")
    anomaly_scores = rx_detector.predict(X_test)
    
    # Evaluate performance
    print("\nEvaluating performance...")
    metrics = evaluate_detector(y_test_binary, anomaly_scores)
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"Average Precision: {metrics['avg_precision']:.3f}")
    
    # Reconstruct 2D image from anomaly scores
    print("\nReconstructing anomaly score map...")
    scores_2d = reconstruct_test_data(anomaly_scores, original_shape, non_background_idx, idx_test)
    y_binary_2d = reconstruct_test_data(y_test_binary, original_shape, non_background_idx, idx_test)
    
    # Save metrics and visualizations to organized directory
    print("\nSaving metrics and visualizations...")
    save_metrics_plots(metrics, scores_2d, y_binary_2d, PROJECT_DIRS['metrics'])
    
    # Save trained model to organized directory
    print("\nSaving trained model...")
    model_path = os.path.join(PROJECT_DIRS['models'], 'rx_detector_model.joblib')
    rx_detector.save_model(model_path)
    
    # Save results
    results_path = os.path.join(PROJECT_DIRS['results'], 'rx_detector_results.npz')
    np.savez(results_path,
             anomaly_scores=anomaly_scores,
             anomaly_scores_2d=scores_2d,
             y_test_binary=y_test_binary,
             y_binary_2d=y_binary_2d,
             metrics=metrics)
    
if __name__ == "__main__":
    main()