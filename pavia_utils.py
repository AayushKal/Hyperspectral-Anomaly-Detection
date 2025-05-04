# pavia_utils.py
# Utility functions for working with preprocessed Pavia University dataset

import numpy as np
import matplotlib.pyplot as plt
import os

def reconstruct_image(predictions, original_shape, non_background_idx):
    """Reconstruct the 2D image from pixel predictions"""
    rows, cols, _ = original_shape
    full_pred = np.zeros(rows * cols)
    full_pred[non_background_idx] = predictions
    return full_pred.reshape(rows, cols)

def visualize_predictions(predictions, original_shape, non_background_idx, 
                         title='Predictions', output_path=None):
    """Visualize prediction results"""
    reconstructed = reconstruct_image(predictions, original_shape, non_background_idx)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(reconstructed, cmap='tab20')
    plt.title(title)
    plt.colorbar()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compare_predictions(y_true, y_pred, original_shape, non_background_idx, 
                       output_dir='outputs', filename_prefix='comparison'):
    """Compare ground truth and predictions side by side"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Reconstruct both
    true_img = reconstruct_image(y_true, original_shape, non_background_idx)
    pred_img = reconstruct_image(y_pred, original_shape, non_background_idx)
    
    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    im1 = ax1.imshow(true_img, cmap='tab20')
    ax1.set_title('Ground Truth')
    ax1.axis('off')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    im2 = ax2.imshow(pred_img, cmap='tab20')
    ax2.set_title('Predictions')
    ax2.axis('off')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{filename_prefix}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create difference map
    diff_img = true_img - pred_img
    
    plt.figure(figsize=(10, 8))
    plt.imshow(diff_img, cmap='coolwarm')
    plt.title('Difference Map (GT - Predictions)')
    plt.colorbar(label='Class Difference')
    
    diff_path = os.path.join(output_dir, f'{filename_prefix}_difference.png')
    plt.savefig(diff_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved to: {output_path}")
    print(f"Difference map saved to: {diff_path}")

def visualize_train_test_split(y_train, y_test, idx_train, idx_test, 
                              original_shape, non_background_idx, 
                              output_dir='outputs'):
    """Visualize the train-test split"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    rows, cols, _ = original_shape
    train_pixels = np.zeros(rows * cols)
    test_pixels = np.zeros(rows * cols)
    
    # Get the indices of non-background pixels
    non_bg_indices = np.where(non_background_idx)[0]
    
    # Assign training and testing labels to their corresponding pixel positions
    train_positions = non_bg_indices[idx_train]
    test_positions = non_bg_indices[idx_test]
    
    train_pixels[train_positions] = y_train
    test_pixels[test_positions] = y_test
    
    # Convert to 2D images
    train_img = train_pixels.reshape(rows, cols)
    test_img = test_pixels.reshape(rows, cols)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    im1 = ax1.imshow(train_img, cmap='tab20')
    ax1.set_title('Training Pixels')
    ax1.axis('off')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    im2 = ax2.imshow(test_img, cmap='tab20')
    ax2.set_title('Testing Pixels')
    ax2.axis('off')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'train_test_split.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Train-test split visualization saved to: {output_path}")

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    from sklearn.metrics import confusion_matrix
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics (weighted for imbalanced classes)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = np.unique(y_true)
    plt.xticks(classes, classes)
    plt.yticks(classes, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }

def load_preprocessed_data(filepath='preprocessed_data/pavia_university_preprocessed.npz'):
    """Load preprocessed data"""
    data = np.load(filepath, allow_pickle=True)
    
    return {
        'X_train': data['X_train'],
        'X_test': data['X_test'],
        'y_train': data['y_train'],
        'y_test': data['y_test'],
        'idx_train': data['idx_train'],
        'idx_test': data['idx_test'],
        'scaler': data['scaler'],
        'original_shape': data['original_shape'],
        'non_background_idx': data['non_background_idx']
    }

def load_reconstruction_params(filepath='preprocessed_data/reconstruction_params.npz'):
    """Load reconstruction parameters"""
    data = np.load(filepath)
    
    return {
        'original_shape': data['original_shape'],
        'non_background_idx': data['non_background_idx']
    }