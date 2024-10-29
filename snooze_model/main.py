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

"""def add_noise(y, noise_factor=0.003):
    noise = np.random.randn(len(y))
    augmented_y = y + noise_factor * noise
    return augmented_y"""

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

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
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
            features = np.zeros((self.n_mels, int(self.fixed_length / (self.sample_rate / 512))), dtype=np.float32)
            label = -1
        else:
            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std
            label = self.labels_dict.get(file_id, -1)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

class AudioClassifier(nn.Module): # inherits from torch.nn.Module, a CNN model
    def __init__(self, num_classes, n_mels, fixed_length, sample_rate):
        super(AudioClassifier, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.flattened_size = self._get_flattened_size(n_mels, fixed_length, sample_rate)
        print(f"Flattened size: {self.flattened_size}")

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def _get_flattened_size(self, n_mels, fixed_length, sample_rate):
        # Create a dummy input tensor with the expected dimensions
        # y is padded/truncated to fixed_length samples
        # mel_spec time dimension ~1287 frames
        # After pooling: ~160 frames
        # To account for possible slight variations, use fixed_length and sample_rate to approximate
        hop_length = 512
        n_fft = 2048
        num_frames = 1 + int((fixed_length - n_fft) / hop_length)  # ~1287
        
        dummy_time_frames = num_frames
        dummy_input = torch.zeros(1, 1, n_mels, dummy_time_frames)
        x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        flattened_size = x.view(1, -1).size(1)
        return flattened_size
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (Batch, 1, N_MELS, Time)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (Batch, 16, N_MELS/2, Time/2)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (Batch, 32, N_MELS/4, Time/4)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> (Batch, 64, N_MELS/8, Time/8)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output logits
        return x

def main():
    audio_path = './assets'
    labels_csv = './labels.csv'
    model_save_path = 'audio_classifier.pth'

    sample_rate = 44100
    n_mels = 128
    fixed_time = 15  # seconds
    fixed_length = sample_rate * fixed_time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    labels_df = pd.read_csv(labels_csv)
    labels_dict = dict(zip(labels_df['file_id'], labels_df['label']))

    # Verify labels are binary
    unique_labels = labels_df['label'].unique()
    if set(unique_labels) != {0, 1}:
        raise ValueError(f"Labels should be binary (0 and 1). Found labels: {unique_labels}")
    print(f"Labels are binary: {unique_labels}")

    train_dataset_full = AudioDataset(audio_path, labels_dict, sample_rate, n_mels, fixed_length, augment=True)
    val_dataset_full = AudioDataset(audio_path, labels_dict, sample_rate, n_mels, fixed_length, augment=False)

    mean, std = compute_global_mean_std(train_dataset_full)
    print(f"Computed global mean: {mean}, std: {std}")
    train_dataset_full.set_normalization(mean, std)
    val_dataset_full.set_normalization(mean, std)

    # checks for empty dataset
    if len(train_dataset_full) == 0:
        raise ValueError("No audio files found or no matching labels in labels.csv.")
    print(f"Dataset size: {len(train_dataset_full)}")


    labels_for_split = [labels_dict[f] for f in train_dataset_full.audio_files]
    train_indices, val_indices = train_test_split(
        list(range(len(train_dataset_full))), test_size=0.2, random_state=42, stratify=labels_for_split
    )
    print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    batch_size = 8

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = 2
    model = AudioClassifier(num_classes, n_mels, fixed_length, sample_rate)

    """class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0,1]),
        y=labels_for_split
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)"""

    criterion = nn.CrossEntropyLoss() # weight=class_weights
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded model and optimizer from {model_save_path}")
    else:
        print("No saved model found, training from scratch")

    model.to(device)

    # Training setup

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            # filter out labels with -1 (corrupted files)
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

        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for features, labels in val_loader:
                # filter out labels with -1 (corrupted files)
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

        if total == 0:
            accuracy = 0
        else:
            accuracy = 100 * correct / total

        if all_labels:
            mae = mean_absolute_error(all_labels, all_predictions)
        else:
            mae = 0
        print(f"Validation Accuracy: {accuracy:.2f}%, MAE: {mae:.7f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_save_path)

if __name__ == '__main__':
    main()