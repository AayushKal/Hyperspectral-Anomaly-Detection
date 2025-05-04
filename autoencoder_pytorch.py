# autoencoder_organized.py
# PyTorch Autoencoder anomaly detection implementation for hyperspectral images

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import os
import time

class HyperspectralAutoencoder(nn.Module):
    """Simple Autoencoder for anomaly detection in hyperspectral data"""
    
    def __init__(self, input_dim=103, encoding_dim=32):
        super(HyperspectralAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderDetector:
    """Autoencoder-based anomaly detector"""
    
    def __init__(self, input_dim=103, encoding_dim=32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = HyperspectralAutoencoder(input_dim, encoding_dim)
        self.is_trained = False
        self.history = {'loss': [], 'val_loss': []}
        
    def train(self, X_train, epochs=100, batch_size=256, validation_split=0.1, learning_rate=0.001):
        """Train the autoencoder on normal data"""
        
        print(f"Training autoencoder on {X_train.shape[0]} samples...")
        print(f"Input dimension: {self.input_dim}")
        print(f"Encoding dimension: {self.encoding_dim}")
        
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X_train)
        
        # Create train-validation split
        val_size = int(len(X_tensor) * validation_split)
        train_size = len(X_tensor) - val_size
        train_data = X_tensor[:train_size]
        val_data = X_tensor[train_size:]
        
        # Create data loaders
        train_dataset = TensorDataset(train_data, train_data)
        val_dataset = TensorDataset(val_data, val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_data, _ in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_data)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['loss'].append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_data, _ in val_loader:
                    outputs = self.model(batch_data)
                    loss = criterion(outputs, batch_data)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.history['val_loss'].append(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Calculate reconstruction errors as anomaly scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get reconstructions
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
        
        # Calculate MSE reconstruction error for each sample
        reconstruction_errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1).numpy()
        
        return reconstruction_errors
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.input_dim = checkpoint['input_dim']
        self.encoding_dim = checkpoint['encoding_dim']
        self.history = checkpoint['history']
        self.is_trained = True
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

def plot_training_history(history, output_dir):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics_plots(metrics, scores_2d, y_binary_2d, output_dir):
    """Save all metric visualizations to output directory"""
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr = metrics['roc_curve']
    plt.plot(fpr, tpr, label=f'ROC AUC = {metrics["roc_auc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Autoencoder')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot PR curve
    plt.figure(figsize=(8, 6))
    precision, recall = metrics['pr_curve']
    plt.plot(recall, precision, label=f'AP = {metrics["avg_precision"]:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Autoencoder')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot anomaly score heatmap
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(scores_2d, cmap='jet')
    plt.colorbar(label='Reconstruction Error')
    plt.title('Reconstruction Error Map')
    
    plt.subplot(1, 2, 2)
    plt.imshow(y_binary_2d, cmap='binary')
    plt.colorbar(label='Ground Truth (1=Anomaly)')
    plt.title('Ground Truth Anomalies')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'anomaly_maps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary metrics text file
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write("Autoencoder Performance Metrics\n")
        f.write("===============================\n\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.3f}\n")
        f.write(f"Average Precision: {metrics['avg_precision']:.3f}\n")
    
    print(f"Metrics and visualizations saved to: {output_dir}")

def main():
    # Create organized directory structure
    PROJECT_DIRS = {
        'preprocessed_data': 'preprocessed_data',
        'models': 'models/autoencoder',
        'metrics': 'outputs/metrics/autoencoder',
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
    
    # Filter training data to only normal samples (non-anomalies)
    normal_mask = y_train_binary == 0
    X_train_normal = X_train[normal_mask]
    print(f"\nNormal training samples: {X_train_normal.shape[0]}")
    
    # Build and train autoencoder
    print("\nBuilding autoencoder...")
    autoencoder = AutoencoderDetector(input_dim=X_train.shape[1], encoding_dim=32)
    
    print("\nTraining autoencoder...")
    autoencoder.train(X_train_normal, epochs=100, batch_size=256)
    
    # Plot training history
    plot_training_history(autoencoder.history, PROJECT_DIRS['metrics'])
    
    # Predict on test data
    print("\nCalculating reconstruction errors...")
    reconstruction_errors = autoencoder.predict(X_test)
    
    # Evaluate performance
    print("\nEvaluating performance...")
    metrics = evaluate_detector(y_test_binary, reconstruction_errors)
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"Average Precision: {metrics['avg_precision']:.3f}")
    
    # Reconstruct 2D image from reconstruction errors
    print("\nReconstructing error map...")
    errors_2d = reconstruct_test_data(reconstruction_errors, original_shape, non_background_idx, idx_test)
    y_binary_2d = reconstruct_test_data(y_test_binary, original_shape, non_background_idx, idx_test)
    
    # Save metrics and visualizations
    print("\nSaving metrics and visualizations...")
    save_metrics_plots(metrics, errors_2d, y_binary_2d, PROJECT_DIRS['metrics'])
    
    # Save trained model
    print("\nSaving trained model...")
    model_path = os.path.join(PROJECT_DIRS['models'], 'autoencoder_model.pth')
    autoencoder.save_model(model_path)
    
    # Save results
    results_path = os.path.join(PROJECT_DIRS['results'], 'autoencoder_results.npz')
    np.savez(results_path,
             reconstruction_errors=reconstruction_errors,
             errors_2d=errors_2d,
             y_test_binary=y_test_binary,
             y_binary_2d=y_binary_2d,
             metrics=metrics)
    
    print("\nAutoencoder anomaly detection complete!")
    print(f"Results saved to: {PROJECT_DIRS['results']}")

if __name__ == "__main__":
    main()