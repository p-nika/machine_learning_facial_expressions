import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FacialExpressionDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform=None, is_test=False):
        self.data = data
        self.transform = transform
        self.is_test = is_test
        
        self.emotion_map = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pixels = self.data.iloc[idx]['pixels']
        image = np.array([int(pixel) for pixel in pixels.split()]).reshape(48, 48)
        
        image = np.stack([image] * 3, axis=-1).astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.FloatTensor(image).unsqueeze(0) / 255.0
        
        if self.is_test:
            return image
        else:
            emotion = int(self.data.iloc[idx]['emotion'])
            return image, emotion

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_data_statistics(train_df: pd.DataFrame) -> dict:
    stats = {}
    
    stats['total_samples'] = len(train_df)
    stats['num_classes'] = train_df['emotion'].nunique()
    
    class_counts = train_df['emotion'].value_counts().sort_index()
    stats['class_distribution'] = class_counts.to_dict()
    
    sample_pixels = []
    for i in range(min(1000, len(train_df))):
        pixels = np.array([int(p) for p in train_df.iloc[i]['pixels'].split()])
        sample_pixels.extend(pixels)
    
    sample_pixels = np.array(sample_pixels)
    stats['pixel_stats'] = {
        'mean': float(np.mean(sample_pixels)),
        'std': float(np.std(sample_pixels)),
        'min': float(np.min(sample_pixels)),
        'max': float(np.max(sample_pixels))
    }
    
    return stats

def visualize_data_distribution(train_df: pd.DataFrame, save_path: Optional[str] = None):
    emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
                   4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    ax1 = axes[0, 0]
    class_counts = train_df['emotion'].value_counts().sort_index()
    bars = ax1.bar(range(len(class_counts)), class_counts.values)
    ax1.set_xlabel('Emotion')
    ax1.set_ylabel('Count')
    ax1.set_title('Class Distribution')
    ax1.set_xticks(range(len(emotion_map)))
    ax1.set_xticklabels([emotion_map[i] for i in range(len(emotion_map))], rotation=45)
    
    for bar, count in zip(bars, class_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(count), ha='center', va='bottom')
    
    for emotion in range(7):
        if emotion == 0:
            continue  # Skip first subplot used for distribution
        
        ax = axes[emotion//4, emotion%4] if emotion < 4 else axes[1, emotion-4]
        
        emotion_samples = train_df[train_df['emotion'] == emotion-1 if emotion > 0 else emotion]
        if len(emotion_samples) > 0:
            pixels = emotion_samples.iloc[0]['pixels']
            image = np.array([int(p) for p in pixels.split()]).reshape(48, 48)
            ax.imshow(image, cmap='gray')
            ax.set_title(f'{emotion_map[emotion-1 if emotion > 0 else emotion]}')
            ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def get_transforms(augment=False):
    if augment:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(mean=[0.485], std=[0.229]),  # ImageNet stats adapted for grayscale
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ])
    
    return transform

def create_data_loaders(train_df: pd.DataFrame, 
                       batch_size: int = 32,
                       val_split: float = 0.2,
                       augment: bool = True,
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
    
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        train_df, test_size=val_split, 
        stratify=train_df['emotion'], 
        random_state=random_state
    )
    
    train_transform = get_transforms(augment=augment)
    val_transform = get_transforms(augment=False)
    
    train_dataset = FacialExpressionDataset(train_data, transform=train_transform)
    val_dataset = FacialExpressionDataset(val_data, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    return train_loader, val_loader

def create_kfold_loaders(train_df: pd.DataFrame, 
                        k_folds: int = 5,
                        batch_size: int = 32,
                        augment: bool = True,
                        random_state: int = 42):
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    fold_loaders = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['emotion'])):
        train_data = train_df.iloc[train_idx].reset_index(drop=True)
        val_data = train_df.iloc[val_idx].reset_index(drop=True)
        
        train_transform = get_transforms(augment=augment)
        val_transform = get_transforms(augment=False)
        
        train_dataset = FacialExpressionDataset(train_data, transform=train_transform)
        val_dataset = FacialExpressionDataset(val_data, transform=val_transform)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=2, pin_memory=True
        )
        
        fold_loaders.append((train_loader, val_loader))
    
    return fold_loaders
