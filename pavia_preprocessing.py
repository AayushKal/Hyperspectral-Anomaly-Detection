# pavia_preprocessing.py
# Preprocessing script for Pavia University Hyperspectral Dataset
# Adapted for VS Code environment

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import os

# Set plotting style
plt.style.use('seaborn-v0_8')

def load_pavia_dataset(data_path='.', data_file='PaviaU.mat', gt_file='PaviaU_gt.mat'):
    """Load Pavia University dataset from local directory"""
    
    # Construct full paths
    data_path_full = os.path.join(data_path, data_file)
    gt_path_full = os.path.join(data_path, gt_file)
    
    # Check if files exist
    if not os.path.exists(data_path_full):
        raise FileNotFoundError(f"Data file not found: {data_path_full}")
    if not os.path.exists(gt_path_full):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path_full}")
    
    # Load the .mat files
    data_dict = sio.loadmat(data_path_full)
    gt_dict = sio.loadmat(gt_path_full)
    
    # Extract data using the standard keys
    data_key = 'paviaU' if 'paviaU' in data_dict else [key for key in data_dict.keys() if not key.startswith('__')][0]
    gt_key = 'paviaU_gt' if 'paviaU_gt' in gt_dict else [key for key in gt_dict.keys() if not key.startswith('__')][0]
    
    data = data_dict[data_key]
    groundtruth = gt_dict[gt_key]
    
    print(f"Data shape: {data.shape}")
    print(f"Groundtruth shape: {groundtruth.shape}")
    print(f"Number of spectral bands: {data.shape[2]}")
    print(f"Unique classes in groundtruth: {np.unique(groundtruth)}")
    
    return data, groundtruth

def visualize_dataset(data, groundtruth, output_dir='outputs'):
    """Visualize the dataset and save plots"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Class names for Pavia University
    class_names = ['Background', 'Asphalt', 'Meadows', 'Gravel', 'Trees', 
                   'Painted metal sheets', 'Bare Soil', 'Bitumen', 
                   'Self-Blocking Bricks', 'Shadows']
    
    # Plot the first band
    plt.figure(figsize=(10, 8))
    plt.imshow(data[:,:,0], cmap='gray')
    plt.title('First Band of Pavia University')
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, 'first_band.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot the groundtruth
    plt.figure(figsize=(10, 8))
    plt.imshow(groundtruth, cmap='tab20')
    plt.title('Groundtruth - Pavia University')
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, 'groundtruth.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Class distribution
    unique_classes, counts = np.unique(groundtruth, return_counts=True)
    df_classes = pd.DataFrame({
        'Class': [class_names[i] if i < len(class_names) else f'Class {i}' 
                  for i in unique_classes],
        'Count': counts
    })
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_classes, x='Class', y='Count')
    plt.xticks(rotation=45, ha='right')
    plt.title('Class Distribution in Pavia University Dataset')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_classes

def preprocess_data(data, groundtruth, test_size=0.3, scaling_method='standard'):
    """Preprocess the hyperspectral data"""
    
    # Reshape data from 3D to 2D (pixels × bands)
    rows, cols, bands = data.shape
    data_2d = data.reshape(rows * cols, bands)
    groundtruth_1d = groundtruth.reshape(rows * cols)
    
    # Remove background pixels (class 0)
    non_background_idx = groundtruth_1d > 0
    X = data_2d[non_background_idx]
    y = groundtruth_1d[non_background_idx]
    
    print(f"Original shape: {data_2d.shape}")
    print(f"After removing background: {X.shape}")
    print(f"Background pixels removed: {data_2d.shape[0] - X.shape[0]}")
    
    # Normalization
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Scaling method must be 'standard' or 'minmax'")
    
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nScaling method: {scaling_method}")
    print(f"Mean of first band (before scaling): {X[:, 0].mean():.3f}")
    print(f"Mean of first band (after scaling): {X_scaled[:, 0].mean():.3f}")
    print(f"Std of first band (before scaling): {X[:, 0].std():.3f}")
    print(f"Std of first band (after scaling): {X_scaled[:, 0].std():.3f}")
    
    # Train-test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, np.arange(len(y)), test_size=test_size, random_state=42, stratify=y)
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Check class distribution in splits
    print("\nClass distribution in training set:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique_train, counts_train):
        print(f"Class {cls}: {count} samples")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'idx_train': idx_train,
        'idx_test': idx_test,
        'scaler': scaler,
        'original_shape': (rows, cols, bands),
        'non_background_idx': non_background_idx
    }

def visualize_spectral_signatures(X_train, y_train, output_dir='outputs'):
    """Plot average spectral signatures for each class"""
    
    # Class names
    class_names = ['Background', 'Asphalt', 'Meadows', 'Gravel', 'Trees', 
                   'Painted metal sheets', 'Bare Soil', 'Bitumen', 
                   'Self-Blocking Bricks', 'Shadows']
    
    plt.figure(figsize=(12, 8))
    for class_id in np.unique(y_train):
        mask = y_train == class_id
        class_spectra = X_train[mask]
        mean_spectrum = np.mean(class_spectra, axis=0)
        plt.plot(mean_spectrum, label=class_names[class_id])
    
    plt.xlabel('Band Number')
    plt.ylabel('Reflectance')
    plt.title('Average Spectral Signatures per Class')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_signatures.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_preprocessed_data(preprocessed_data, output_path='preprocessed_data'):
    """Save preprocessed data to disk"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save as numpy archive
    output_file = os.path.join(output_path, 'pavia_university_preprocessed.npz')
    np.savez(output_file, **preprocessed_data)
    
    # Save reconstruction parameters separately for easy loading
    reconstruction_file = os.path.join(output_path, 'reconstruction_params.npz')
    np.savez(reconstruction_file,
             original_shape=preprocessed_data['original_shape'],
             non_background_idx=preprocessed_data['non_background_idx'])
    
    print(f"\nPreprocessed data saved to: {output_file}")
    print(f"Reconstruction parameters saved to: {reconstruction_file}")
    
    return output_file

def main():
    """Main preprocessing pipeline"""
    
    print("=== Pavia University Dataset Preprocessing ===\n")
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    try:
        # 1. Load dataset
        print("1. Loading dataset...")
        data, groundtruth = load_pavia_dataset()
        
        # 2. Visualize dataset
        print("\n2. Visualizing dataset...")
        df_classes = visualize_dataset(data, groundtruth)
        print(df_classes)
        
        # 3. Preprocess data
        print("\n3. Preprocessing data...")
        preprocessed_data = preprocess_data(data, groundtruth)
        
        # 4. Visualize spectral signatures
        print("\n4. Visualizing spectral signatures...")
        visualize_spectral_signatures(preprocessed_data['X_train'], 
                                     preprocessed_data['y_train'])
        
        # 5. Save preprocessed data
        print("\n5. Saving preprocessed data...")
        output_file = save_preprocessed_data(preprocessed_data)
        
        print("\n=== Preprocessing Complete! ===")
        print(f"All outputs saved to 'outputs' directory")
        print(f"Preprocessed data saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()