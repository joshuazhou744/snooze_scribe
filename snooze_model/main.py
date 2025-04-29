import os
import librosa
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_absolute_error

def add_noise(y, noise_factor=0.003):
    noise = np.random.randn(len(y))
    augmented_y = y + noise_factor * noise
    return augmented_y

def custom_time_stretch(y, rate=1.1):
    return librosa.effects.time_stretch(y, rate=rate)

def extract_features(file_path, sample_rate, n_mels, fixed_length, augment=False):
    try:
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True)

        if augment: # for training only
            y = custom_time_stretch(y, rate=1.1)

        if len(y) < fixed_length:
            padding = fixed_length - len(y)
            y = np.pad(y, (0, padding), 'constant')
        else:
            y = y[:fixed_length]

        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels,
            hop_length=1024  # Larger hop length = fewer time frames = smaller model
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return log_mel_spec
    except Exception as e: 
        print(f"Error processing {file_path}", e)
        return None
    
def compute_global_mean_std(dataset):
    from tqdm import tqdm
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    all_features = []
    print("Computing global mean and std...")
    for features, _ in tqdm(loader):
        all_features.append(features)
    all_features = torch.cat(all_features, dim=0)  # Shape: (num_samples, n_mels, time_steps)
    mean = all_features.mean()
    std = all_features.std()
    return mean.item(), std.item()

class AudioDataset(Dataset):
    def __init__(self, audio_path, labels_dict, sample_rate, n_mels, fixed_length, augment=False, mean=None, std=None):
        self.audio_path = audio_path
        self.labels_dict = labels_dict
        self.augment = augment
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fixed_length = fixed_length
        self.audio_files = [f for f in os.listdir(audio_path) if f.endswith('.mp4')]
        self.audio_files = [f for f in self.audio_files if f in labels_dict]
        print(f"Total audio files found: {len(self.audio_files)}")
        self.mean = mean
        self.std = std

    def set_normalization(self, mean, std):
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, i):
        file_id = self.audio_files[i]
        file_path = os.path.join(self.audio_path, file_id)
        features = extract_features(file_path, self.sample_rate, self.n_mels, self.fixed_length, augment=self.augment)
        label = self.labels_dict.get(file_id, -1)
        if features is None:
            # Handle corrupted file by returning a zero tensor and label -1
            features = np.zeros((self.n_mels, int(self.fixed_length / (self.sample_rate / 1024))), dtype=np.float32)
            label = -1
        else:
            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std
            label = self.labels_dict.get(file_id, -1)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

class EfficientAudioClassifier(nn.Module):
    """Lightweight CNN model for audio classification with fewer parameters"""
    def __init__(self, num_classes, n_mels, fixed_length, sample_rate):
        super(EfficientAudioClassifier, self).__init__()
        
        hop_length = 1024
        self.time_dim = int(fixed_length / hop_length) + 1
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv_dw = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)
        self.bn_dw = nn.BatchNorm2d(16)
        self.conv_pw = nn.Conv2d(16, 32, kernel_size=1)
        self.bn_pw = nn.BatchNorm2d(32)
        
        n_mels_reduced = n_mels // 4
        time_dim_reduced = self.time_dim // 4
        self.flattened_size = 32 * n_mels_reduced * time_dim_reduced
        
        print(f"Flattened size: {self.flattened_size}")
        
        # Smaller fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 64)  # Reduced from 128 to 64
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Add channel dimension [batch, features, time] -> [batch, 1, features, time]
        x = x.unsqueeze(1)
        
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Efficient depthwise separable convolution block
        x = F.relu(self.bn_dw(self.conv_dw(x)))
        x = self.pool(F.relu(self.bn_pw(self.conv_pw(x))))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def main():
    audio_path = './train_data'
    labels_csv = './labels.csv'
    model_save_path = 'audio_classifier.pth'

    # Audio processing parameters
    sample_rate = 44100
    n_mels = 64  # Reduced from 128 to 64 mel bands for efficiency
    fixed_time = 15  # seconds
    fixed_length = sample_rate * fixed_time

    # Set device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and verify labels
    labels_df = pd.read_csv(labels_csv)
    labels_dict = dict(zip(labels_df['file_id'], labels_df['label']))

    # Verify labels are binary
    unique_labels = labels_df['label'].unique()
    if set(unique_labels) != {0, 1}:
        raise ValueError(f"Labels should be binary (0 and 1). Found labels: {unique_labels}")
    print(f"Labels are binary: {unique_labels}")

    # Create datasets
    train_dataset_full = AudioDataset(audio_path, labels_dict, sample_rate, n_mels, fixed_length, augment=True)
    val_dataset_full = AudioDataset(audio_path, labels_dict, sample_rate, n_mels, fixed_length, augment=False)

    # Compute normalization statistics
    mean, std = compute_global_mean_std(train_dataset_full)
    print(f"Computed global mean: {mean}, std: {std}")
    train_dataset_full.set_normalization(mean, std)
    val_dataset_full.set_normalization(mean, std)

    # Check for empty dataset
    if len(train_dataset_full) == 0:
        raise ValueError("No audio files found or no matching labels in labels.csv.")
    print(f"Dataset size: {len(train_dataset_full)}")

    # Create train/validation split
    labels_for_split = [labels_dict[f] for f in train_dataset_full.audio_files]
    train_indices, val_indices = train_test_split(
        list(range(len(train_dataset_full))), test_size=0.2, random_state=42, stratify=labels_for_split
    )
    print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    # Create data loaders
    batch_size = 16  # Increased batch size for faster training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize the efficient model
    num_classes = 2
    model = EfficientAudioClassifier(num_classes, n_mels, fixed_length, sample_rate)
    model.to(device)
    
    # Print model summary and parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay

    # Load existing model if available
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded model and optimizer from {model_save_path}")
    else:
        print("No saved model found, training from scratch")

    # Training loop
    num_epochs = 15  # Increased epochs for better learning

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            # Filter out labels with -1 (corrupted files)
            valid_indices = labels != -1
            if not valid_indices.any():
                continue
            features = features[valid_indices]
            labels = labels[valid_indices]

            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for features, labels in val_loader:
                # Filter out labels with -1 (corrupted files)
                valid_indices = labels != -1
                if not valid_indices.any():
                    continue
                features = features[valid_indices]
                labels = labels[valid_indices]

                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Calculate accuracy and MAE
        if total == 0:
            accuracy = 0
        else:
            accuracy = 100 * correct / total

        if all_labels:
            mae = mean_absolute_error(all_labels, all_predictions)
        else:
            mae = 0
        print(f"Validation Accuracy: {accuracy:.2f}%, MAE: {mae:.7f}")

    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_save_path)
    
    print(f"Model saved to {model_save_path}")
    print(f"Model size: {os.path.getsize(model_save_path) / (1024 * 1024):.2f} MB")

if __name__ == '__main__':
    main()