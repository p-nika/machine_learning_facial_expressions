import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import wandb
from typing import Dict, List, Tuple, Optional
import pandas as pd

class ModelEvaluator:
    
    def __init__(self, model: torch.nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.emotion_map = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        self.class_names = list(self.emotion_map.values())
    
    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 2:  # Training data with labels
                    data, targets = batch
                    data, targets = data.to(self.device), targets.to(self.device)
                    all_targets.extend(targets.cpu().numpy())
                else:  # Test data without labels
                    data = batch.to(self.device)
                
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        targets = np.array(all_targets) if all_targets else None
        
        return predictions, probabilities, targets
    
    def evaluate(self, data_loader, log_to_wandb: bool = False) -> Dict:
        predictions, probabilities, targets = self.predict(data_loader)
        
        if targets is None:
            return {'predictions': predictions, 'probabilities': probabilities}
        
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None
        )
        
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        cm = confusion_matrix(targets, predictions)
        
        report = classification_report(
            targets, predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'support': support,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions,
            'probabilities': probabilities,
            'targets': targets
        }
        
        try:
            targets_binary = label_binarize(targets, classes=range(7))
            if targets_binary.shape[1] > 1:  # Multiclass
                auc_scores = []
                for i in range(7):
                    if len(np.unique(targets_binary[:, i])) > 1:  # Class present in data
                        auc = roc_auc_score(targets_binary[:, i], probabilities[:, i])
                        auc_scores.append(auc)
                    else:
                        auc_scores.append(0.0)
                results['per_class_auc'] = np.array(auc_scores)
                results['macro_auc'] = np.mean(auc_scores)
        except Exception as e:
            print(f"Could not compute AUC: {e}")
            results['per_class_auc'] = np.zeros(7)
            results['macro_auc'] = 0.0
        
        if log_to_wandb:
            wandb.log({
                'test_accuracy': accuracy,
                'test_macro_f1': macro_f1,
                'test_weighted_f1': weighted_f1,
                'test_macro_auc': results['macro_auc'],
                'confusion_matrix': wandb.plot.confusion_matrix(
                    probs=None, y_true=targets, preds=predictions,
                    class_names=self.class_names
                )
            })
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, normalize: bool = False, 
                            save_path: Optional[str] = None):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_class_metrics(self, results: Dict, save_path: Optional[str] = None):
        metrics = ['per_class_precision', 'per_class_recall', 'per_class_f1']
        metric_names = ['Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            scores = results[metric]
            bars = axes[i].bar(self.class_names, scores)
            axes[i].set_title(f'Per-Class {name}')
            axes[i].set_ylabel(name)
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_misclassifications(self, results: Dict, data_loader, 
                                 num_examples: int = 5) -> Dict:
        predictions = results['predictions']
        targets = results['targets']
        probabilities = results['probabilities']
        
        misclassified_indices = np.where(predictions != targets)[0]
        
        misclassified_confidences = []
        for idx in misclassified_indices:
            predicted_class = predictions[idx]
            confidence = probabilities[idx, predicted_class]
            misclassified_confidences.append(confidence)
        
        sorted_indices = np.argsort(misclassified_confidences)[::-1]
        
        analysis = {
            'total_misclassified': len(misclassified_indices),
            'misclassification_rate': len(misclassified_indices) / len(targets),
            'high_confidence_mistakes': []
        }
        
        for i in range(min(num_examples, len(sorted_indices))):
            idx = misclassified_indices[sorted_indices[i]]
            analysis['high_confidence_mistakes'].append({
                'index': idx,
                'true_label': self.emotion_map[targets[idx]],
                'predicted_label': self.emotion_map[predictions[idx]],
                'confidence': misclassified_confidences[sorted_indices[i]],
                'all_probabilities': probabilities[idx]
            })
        
        return analysis
