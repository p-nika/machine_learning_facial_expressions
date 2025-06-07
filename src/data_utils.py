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
import json
import os

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

def load_split_data(data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    train_path = os.path.join(data_dir, 'train_split.csv')
    val_path = os.path.join(data_dir, 'val_split.csv')
    test_path = os.path.join(data_dir, 'test_split.csv')
    
    if not all(os.path.exists(path) for path in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            "Split files not found! Please run the updated data exploration notebook first "
            "to create train_split.csv, val_split.csv, and test_split.csv"
        )
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, val_df, test_df

def load_split_metadata(data_dir: str = 'data') -> dict:
    metadata_path = os.path.join(data_dir, 'split_metadata.json')
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    else:
        return {
            'split_ratios': {'train': 0.7, 'val': 0.15, 'test': 0.15},
            'random_state': 42,
            'stratified': True
        }

def verify_split_integrity(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    
    train_pixels = set(train_df['pixels'].values)
    val_pixels = set(val_df['pixels'].values)
    test_pixels = set(test_df['pixels'].values)
    
    train_val_overlap = len(train_pixels.intersection(val_pixels))
    train_test_overlap = len(train_pixels.intersection(test_pixels))
    val_test_overlap = len(val_pixels.intersection(test_pixels))
    
    total_overlap = train_val_overlap + train_test_overlap + val_test_overlap
    
    from scipy.stats import chi2_contingency
    
    emotions = range(7)
    train_counts = [len(train_df[train_df['emotion'] == e]) for e in emotions]
    val_counts = [len(val_df[val_df['emotion'] == e]) for e in emotions]
    test_counts = [len(test_df[test_df['emotion'] == e]) for e in emotions]
    
    contingency_table = np.array([train_counts, val_counts, test_counts])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    return total_overlap == 0 and p_value > 0.05

def get_data_statistics(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    
    def analyze_split(df, split_name):
        stats = {
            'total_samples': len(df),
            'num_classes': df['emotion'].nunique()
        }
        
        class_counts = df['emotion'].value_counts().sort_index()
        stats['class_distribution'] = class_counts.to_dict()
        stats['class_percentages'] = (class_counts / len(df) * 100).to_dict()
        
        stats['imbalance_ratio'] = class_counts.max() / class_counts.min()
        
        sample_pixels = []
        sample_size = min(500, len(df))
        
        for i in range(sample_size):
            pixels = np.array([int(p) for p in df.iloc[i]['pixels'].split()])
            sample_pixels.extend(pixels)
        
        sample_pixels = np.array(sample_pixels)
        stats['pixel_stats'] = {
            'mean': float(np.mean(sample_pixels)),
            'std': float(np.std(sample_pixels)),
            'min': float(np.min(sample_pixels)),
            'max': float(np.max(sample_pixels))
        }
        
        return stats
    
    all_stats = {
        'train': analyze_split(train_df, 'train'),
        'validation': analyze_split(val_df, 'validation'),
        'test': analyze_split(test_df, 'test')
    }
    
    total_samples = len(train_df) + len(val_df) + len(test_df)
    all_stats['combined'] = {
        'total_samples': total_samples,
        'split_ratios': {
            'train': len(train_df) / total_samples,
            'validation': len(val_df) / total_samples,
            'test': len(test_df) / total_samples
        }
    }
    
    return all_stats

def visualize_split_distributions(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                                save_path: Optional[str] = None):
    
    emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
                   4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    split_sizes = [len(train_df), len(val_df), len(test_df)]
    split_labels = ['Train', 'Validation', 'Test']
    colors = ['skyblue', 'lightgreen', 'coral']
    
    bars = axes[0, 0].bar(split_labels, split_sizes, color=colors)
    axes[0, 0].set_title('Dataset Split Sizes')
    axes[0, 0].set_ylabel('Number of Samples')
    
    for bar, size in zip(bars, split_sizes):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                       f'{size:,}', ha='center', va='bottom', fontweight='bold')
    
    emotions = list(range(7))
    train_counts = [len(train_df[train_df['emotion'] == e]) for e in emotions]
    val_counts = [len(val_df[val_df['emotion'] == e]) for e in emotions]
    test_counts = [len(test_df[test_df['emotion'] == e]) for e in emotions]
    
    x = np.arange(len(emotions))
    width = 0.25
    
    axes[0, 1].bar(x - width, train_counts, width, label='Train', color='skyblue', alpha=0.8)
    axes[0, 1].bar(x, val_counts, width, label='Validation', color='lightgreen', alpha=0.8)
    axes[0, 1].bar(x + width, test_counts, width, label='Test', color='coral', alpha=0.8)
    
    axes[0, 1].set_xlabel('Emotion')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Class Distribution Across Splits')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([emotion_map[e] for e in emotions], rotation=45)
    axes[0, 1].legend()
    
    train_percentages = np.array(train_counts) / len(train_df) * 100
    val_percentages = np.array(val_counts) / len(val_df) * 100
    test_percentages = np.array(test_counts) / len(test_df) * 100
    
    axes[1, 0].bar(x - width, train_percentages, width, label='Train', color='skyblue', alpha=0.8)
    axes[1, 0].bar(x, val_percentages, width, label='Validation', color='lightgreen', alpha=0.8)
    axes[1, 0].bar(x + width, test_percentages, width, label='Test', color='coral', alpha=0.8)
    
    axes[1, 0].set_xlabel('Emotion')
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].set_title('Class Distribution (Percentages)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([emotion_map[e] for e in emotions], rotation=45)
    axes[1, 0].legend()
    
    train_imbalance = max(train_counts) / min(train_counts)
    val_imbalance = max(val_counts) / min(val_counts)
    test_imbalance = max(test_counts) / min(test_counts)
    
    imbalance_ratios = [train_imbalance, val_imbalance, test_imbalance]
    bars = axes[1, 1].bar(split_labels, imbalance_ratios, color=colors, alpha=0.8)
    axes[1, 1].set_title('Class Imbalance Ratio by Split')
    axes[1, 1].set_ylabel('Imbalance Ratio (Max/Min)')
    axes[1, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Perfect Balance')
    
    for bar, ratio in zip(bars, imbalance_ratios):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def get_transforms(augment=False):
    if augment:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.CoarseDropout(max_holes=1, max_height=8, max_width=8, 
                           min_holes=1, min_height=4, min_width=4, p=0.2),
            A.Normalize(mean=[0.485], std=[0.229]),  # ImageNet stats adapted for grayscale
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ])
    
    return transform

def create_split_data_loaders(data_dir: str = 'data', 
                             batch_size: int = 32,
                             augment_train: bool = True,
                             num_workers: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_df, val_df, test_df = load_split_data(data_dir)
    verify_split_integrity(train_df, val_df, test_df)
    
    train_transform = get_transforms(augment=augment_train)
    val_test_transform = get_transforms(augment=False)
    
    train_dataset = FacialExpressionDataset(train_df, transform=train_transform)
    val_dataset = FacialExpressionDataset(val_df, transform=val_test_transform)
    test_dataset = FacialExpressionDataset(test_df, transform=val_test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def create_kfold_loaders_from_train(data_dir: str = 'data',
                                   k_folds: int = 5,
                                   batch_size: int = 32,
                                   augment: bool = True,
                                   random_state: int = 42):
    
    train_df, _, _ = load_split_data(data_dir)
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    
    fold_loaders = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['emotion'])):
        fold_train_data = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_data = train_df.iloc[val_idx].reset_index(drop=True)
        
        train_transform = get_transforms(augment=augment)
        val_transform = get_transforms(augment=False)
        
        fold_train_dataset = FacialExpressionDataset(fold_train_data, transform=train_transform)
        fold_val_dataset = FacialExpressionDataset(fold_val_data, transform=val_transform)
        
        fold_train_loader = DataLoader(
            fold_train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=2, pin_memory=True
        )
        fold_val_loader = DataLoader(
            fold_val_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=2, pin_memory=True
        )
        
        fold_loaders.append((fold_train_loader, fold_val_loader))
    
    return fold_loaders

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

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
