import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import logging
import os
import json
from typing import Dict, Tuple, List
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive evaluation toolkit for OncoGraph models.
    Generates metrics, visualizations, and statistical analyses.
    """
    
    def __init__(self, model, graph_data, config, device='cpu'):
        self.model = model
        self.graph_data = graph_data.to(device)
        self.config = config
        self.device = device
        
        # Results storage
        self.results = {}
        self.predictions = {}
        
        # Create results directory
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
        
        logger.info("Evaluator initialized")
    
    @torch.no_grad()
    def get_predictions(self, mask_name='test'):
        """
        Get predictions for specified data split.
        
        Args:
            mask_name: 'train', 'val', or 'test'
        
        Returns:
            Dictionary with predictions, true labels, probabilities, embeddings
        """
        self.model.eval()
        
        # Get mask
        if mask_name == 'train':
            mask = self.graph_data.train_mask
        elif mask_name == 'val':
            mask = self.graph_data.val_mask
        elif mask_name == 'test':
            mask = self.graph_data.test_mask
        else:
            raise ValueError(f"Unknown mask: {mask_name}")
        
        # Forward pass
        logits, embeddings = self.model(
            self.graph_data.x,
            self.graph_data.edge_index,
            self.graph_data.edge_attr.squeeze(),
            return_embeddings=True
        )
        
        # Get predictions and probabilities
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        confidences = probs.max(dim=1)[0]
        
        # Extract masked data
        results = {
            'predictions': preds[mask].cpu().numpy(),
            'true_labels': self.graph_data.y[mask].cpu().numpy(),
            'probabilities': probs[mask].cpu().numpy(),
            'confidences': confidences[mask].cpu().numpy(),
            'embeddings': embeddings[mask].cpu().numpy(),
            'logits': logits[mask].cpu().numpy()
        }
        
        self.predictions[mask_name] = results
        
        return results
    
    def compute_basic_metrics(self, mask_name='test') -> Dict:
        """Compute accuracy, precision, recall, F1."""
        
        if mask_name not in self.predictions:
            self.get_predictions(mask_name)
        
        preds = self.predictions[mask_name]['predictions']
        labels = self.predictions[mask_name]['true_labels']
        
        # Overall metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support_per_class': support_per_class.tolist()
        }
        
        self.results[f'{mask_name}_metrics'] = metrics
        
        logger.info(f"{mask_name.upper()} Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        return metrics
    
    def compute_confusion_matrix(self, mask_name='test', normalize=True):
        """Compute and visualize confusion matrix."""
        
        if mask_name not in self.predictions:
            self.get_predictions(mask_name)
        
        preds = self.predictions[mask_name]['predictions']
        labels = self.predictions[mask_name]['true_labels']
        
        # Compute confusion matrix
        cm = confusion_matrix(labels, preds)
        
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        else:
            cm_normalized = cm
        
        self.results[f'{mask_name}_confusion_matrix'] = cm
        
        return cm, cm_normalized
    
    def plot_confusion_matrix(self, mask_name='test', figsize=(20, 16)):
        """Plot confusion matrix heatmap."""
        
        cm, cm_normalized = self.compute_confusion_matrix(mask_name, normalize=True)
        
        # Get class names (assuming they're stored in processed data)
        num_classes = cm.shape[0]
        class_names = [f'C{i}' for i in range(num_classes)]  # Placeholder
        
        # Try to get actual cancer type names
        try:
            from backend.utils.data_processor import DataProcessor
            processor = DataProcessor(self.config)
            processed_data = processor.load_processed_data(self.config.PROCESSED_DATA_PATH)
            class_names = processed_data['cancer_types']
        except:
            pass
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot normalized confusion matrix
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Proportion'}, ax=ax1)
        ax1.set_title(f'Normalized Confusion Matrix ({mask_name.upper()})', fontsize=16)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.tick_params(axis='both', labelsize=8)
        
        # Plot raw counts
        sns.heatmap(cm, annot=False, fmt='d', cmap='Greens',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, ax=ax2)
        ax2.set_title(f'Confusion Matrix - Raw Counts ({mask_name.upper()})', fontsize=16)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.tick_params(axis='both', labelsize=8)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.config.RESULTS_PATH, f'{mask_name}_confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
        
        return fig
    
    def plot_per_class_metrics(self, mask_name='test', figsize=(16, 10)):
        """Plot per-class precision, recall, F1 scores."""
        
        if f'{mask_name}_metrics' not in self.results:
            self.compute_basic_metrics(mask_name)
        
        metrics = self.results[f'{mask_name}_metrics']
        
        # Get class names
        try:
            from backend.utils.data_processor import DataProcessor
            processor = DataProcessor(self.config)
            processed_data = processor.load_processed_data(self.config.PROCESSED_DATA_PATH)
            class_names = processed_data['cancer_types']
        except:
            class_names = [f'C{i}' for i in range(len(metrics['f1_per_class']))]
        
        # Prepare data
        precision = metrics['precision_per_class']
        recall = metrics['recall_per_class']
        f1 = metrics['f1_per_class']
        support = metrics['support_per_class']
        
        # Create DataFrame
        df = pd.DataFrame({
            'Cancer Type': class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        
        # Sort by F1 score
        df = df.sort_values('F1-Score', ascending=True)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: F1 scores
        axes[0, 0].barh(df['Cancer Type'], df['F1-Score'], color='skyblue')
        axes[0, 0].set_xlabel('F1-Score', fontsize=12)
        axes[0, 0].set_title(f'F1-Score by Cancer Type ({mask_name.upper()})', fontsize=14)
        axes[0, 0].set_xlim([0, 1])
        axes[0, 0].axvline(x=df['F1-Score'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["F1-Score"].mean():.3f}')
        axes[0, 0].legend()
        
        # Plot 2: Precision vs Recall
        axes[0, 1].scatter(df['Recall'], df['Precision'], s=df['Support']*2, 
                          alpha=0.6, c=range(len(df)), cmap='viridis')
        axes[0, 1].set_xlabel('Recall', fontsize=12)
        axes[0, 1].set_ylabel('Precision', fontsize=12)
        axes[0, 1].set_title('Precision vs Recall', fontsize=14)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.3)
        axes[0, 1].set_xlim([0, 1])
        axes[0, 1].set_ylim([0, 1])
        
        # Plot 3: Support distribution
        axes[1, 0].bar(range(len(df)), df['Support'], color='lightcoral')
        axes[1, 0].set_xlabel('Cancer Type Index', fontsize=12)
        axes[1, 0].set_ylabel('Sample Count', fontsize=12)
        axes[1, 0].set_title('Support Distribution', fontsize=14)
        
        # Plot 4: Metric comparison
        x = np.arange(len(df))
        width = 0.25
        axes[1, 1].barh(x - width, df['Precision'], width, label='Precision', alpha=0.8)
        axes[1, 1].barh(x, df['Recall'], width, label='Recall', alpha=0.8)
        axes[1, 1].barh(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
        axes[1, 1].set_yticks(x)
        axes[1, 1].set_yticklabels(df['Cancer Type'], fontsize=8)
        axes[1, 1].set_xlabel('Score', fontsize=12)
        axes[1, 1].set_title('Metrics Comparison', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].set_xlim([0, 1])
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.config.RESULTS_PATH, f'{mask_name}_per_class_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Per-class metrics plot saved to {save_path}")
        
        plt.close()
        
        # Save DataFrame to CSV
        csv_path = os.path.join(self.config.RESULTS_PATH, f'{mask_name}_per_class_metrics.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Per-class metrics saved to {csv_path}")
        
        return df
    
    def compute_roc_auc(self, mask_name='test'):
        """Compute ROC-AUC for multi-class classification."""
        
        if mask_name not in self.predictions:
            self.get_predictions(mask_name)
        
        labels = self.predictions[mask_name]['true_labels']
        probs = self.predictions[mask_name]['probabilities']
        
        # Get number of classes
        n_classes = probs.shape[1]
        
        # Binarize labels for multi-class ROC
        labels_bin = label_binarize(labels, classes=range(n_classes))
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(labels_bin.ravel(), probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC AUC
        roc_auc["macro"] = np.mean(list(roc_auc.values()))
        
        self.results[f'{mask_name}_roc'] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }
        
        logger.info(f"ROC-AUC ({mask_name.upper()}):")
        logger.info(f"  Micro-average: {roc_auc['micro']:.4f}")
        logger.info(f"  Macro-average: {roc_auc['macro']:.4f}")
        
        return fpr, tpr, roc_auc
    
    def plot_roc_curves(self, mask_name='test', figsize=(14, 10)):
        """Plot ROC curves for all classes."""
        
        if f'{mask_name}_roc' not in self.results:
            self.compute_roc_auc(mask_name)
        
        roc_data = self.results[f'{mask_name}_roc']
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        roc_auc = roc_data['auc']
        
        # Get class names
        try:
            from backend.utils.data_processor import DataProcessor
            processor = DataProcessor(self.config)
            processed_data = processor.load_processed_data(self.config.PROCESSED_DATA_PATH)
            class_names = processed_data['cancer_types']
        except:
            class_names = [f'C{i}' for i in range(len(roc_auc) - 2)]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: All classes
        n_classes = len(class_names)
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_classes))
        
        for i, color in zip(range(n_classes), colors):
            ax1.plot(fpr[i], tpr[i], color=color, lw=1, alpha=0.3,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        # Plot micro and macro averages
        ax1.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-avg (AUC = {roc_auc["micro"]:.2f})',
                color='deeppink', linestyle='--', linewidth=3)
        
        ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title(f'ROC Curves - All Classes ({mask_name.upper()})', fontsize=14)
        ax1.legend(loc="lower right", fontsize=6, ncol=2)
        ax1.grid(alpha=0.3)
        
        # Plot 2: Top 10 classes by AUC
        class_aucs = [(i, roc_auc[i], class_names[i]) for i in range(n_classes)]
        class_aucs_sorted = sorted(class_aucs, key=lambda x: x[1], reverse=True)[:10]
        
        for idx, auc_val, name in class_aucs_sorted:
            ax2.plot(fpr[idx], tpr[idx], lw=2,
                    label=f'{name} (AUC = {auc_val:.3f})')
        
        ax2.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-avg (AUC = {roc_auc["micro"]:.2f})',
                color='deeppink', linestyle='--', linewidth=3)
        
        ax2.plot([0, 1], [0, 1], 'k--', lw=2)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title(f'ROC Curves - Top 10 Classes ({mask_name.upper()})', fontsize=14)
        ax2.legend(loc="lower right", fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.config.RESULTS_PATH, f'{mask_name}_roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
        
        plt.close()
        
        return fig
    
    def analyze_misclassifications(self, mask_name='test', top_n=10):
        """Analyze most common misclassification patterns."""
        
        if mask_name not in self.predictions:
            self.get_predictions(mask_name)
        
        preds = self.predictions[mask_name]['predictions']
        labels = self.predictions[mask_name]['true_labels']
        confidences = self.predictions[mask_name]['confidences']
        
        # Get class names
        try:
            from backend.utils.data_processor import DataProcessor
            processor = DataProcessor(self.config)
            processed_data = processor.load_processed_data(self.config.PROCESSED_DATA_PATH)
            class_names = processed_data['cancer_types']
        except:
            class_names = [f'C{i}' for i in range(max(labels.max(), preds.max()) + 1)]
        
        # Find misclassifications
        misclassified_mask = preds != labels
        misclassified_indices = np.where(misclassified_mask)[0]
        
        logger.info(f"Misclassifications ({mask_name.upper()}):")
        logger.info(f"  Total: {len(misclassified_indices)} / {len(labels)} "
                   f"({100*len(misclassified_indices)/len(labels):.2f}%)")
        
        # Analyze confusion pairs
        confusion_pairs = defaultdict(int)
        for idx in misclassified_indices:
            true_label = labels[idx]
            pred_label = preds[idx]
            confusion_pairs[(true_label, pred_label)] += 1
        
        # Get top N confusion pairs
        top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        logger.info(f"\n  Top {top_n} Confusion Pairs:")
        for (true_idx, pred_idx), count in top_confusions:
            logger.info(f"    {class_names[true_idx]} → {class_names[pred_idx]}: {count}")
        
        # Analyze confidence on misclassifications
        misclassified_confidences = confidences[misclassified_mask]
        correct_confidences = confidences[~misclassified_mask]
        
        logger.info(f"\n  Confidence Analysis:")
        logger.info(f"    Correct predictions - Mean: {correct_confidences.mean():.3f}, "
                   f"Median: {np.median(correct_confidences):.3f}")
        logger.info(f"    Wrong predictions - Mean: {misclassified_confidences.mean():.3f}, "
                   f"Median: {np.median(misclassified_confidences):.3f}")
        
        return top_confusions, misclassified_indices
    
    def generate_classification_report(self, mask_name='test'):
        """Generate comprehensive classification report."""
        
        if mask_name not in self.predictions:
            self.get_predictions(mask_name)
        
        preds = self.predictions[mask_name]['predictions']
        labels = self.predictions[mask_name]['true_labels']
        
        # Get class names
        try:
            from backend.utils.data_processor import DataProcessor
            processor = DataProcessor(self.config)
            processed_data = processor.load_processed_data(self.config.PROCESSED_DATA_PATH)
            class_names = processed_data['cancer_types']
        except:
            class_names = None
        
        # Generate report
        report = classification_report(
            labels, preds,
            target_names=class_names,
            digits=4,
            output_dict=True
        )
        
        # Save to JSON
        report_path = os.path.join(self.config.RESULTS_PATH, f'{mask_name}_classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Classification report saved to {report_path}")
        
        # Print summary
        logger.info(f"\nClassification Report ({mask_name.upper()}):")
        logger.info(f"  Accuracy: {report['accuracy']:.4f}")
        logger.info(f"  Macro Avg - Precision: {report['macro avg']['precision']:.4f}, "
                   f"Recall: {report['macro avg']['recall']:.4f}, "
                   f"F1: {report['macro avg']['f1-score']:.4f}")
        logger.info(f"  Weighted Avg - Precision: {report['weighted avg']['precision']:.4f}, "
                   f"Recall: {report['weighted avg']['recall']:.4f}, "
                   f"F1: {report['weighted avg']['f1-score']:.4f}")
        
        return report
    
    def run_complete_evaluation(self, mask_name='test'):
        """Run all evaluation metrics and generate visualizations."""
        
        logger.info("="*60)
        logger.info(f"Running Complete Evaluation on {mask_name.upper()} Set")
        logger.info("="*60)
        
        # Get predictions
        self.get_predictions(mask_name)
        
        # Compute metrics
        self.compute_basic_metrics(mask_name)
        self.compute_roc_auc(mask_name)
        
        # Generate visualizations
        self.plot_confusion_matrix(mask_name)
        self.plot_per_class_metrics(mask_name)
        self.plot_roc_curves(mask_name)
        
        # Analysis
        self.analyze_misclassifications(mask_name)
        self.generate_classification_report(mask_name)
        
        logger.info("="*60)
        logger.info("Evaluation Complete")
        logger.info("="*60)
        
        return self.results