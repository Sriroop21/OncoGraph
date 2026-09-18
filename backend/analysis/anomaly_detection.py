import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
import logging
import os
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')


class AnomalyDetectionProV3:
    """
    Professional Anomaly Detection V3 - OVERFITTING FIX.
    
    Key Improvements:
    1. Calibrated thresholds using validation set (prevents train overfitting)
    2. Meta-learned ensemble weights optimized for error detection
    3. Adaptive per-sample anomaly scores (not binary)
    4. Cross-dataset class statistics (uses train+val for better generalization)
    5. Error-focused optimization (maximizes error detection rate)
    
    Target: 75-85% error detection on test, <5% false positive rate
    """
    
    def __init__(self, model, graph_data, config, device='cpu'):
        self.model = model
        self.graph_data = graph_data.to(device)
        self.config = config
        self.device = device
        
        # Will be calibrated on validation set
        self.calibrated_thresholds = None
        self.calibrated_weights = None
        
        # Class statistics
        self.class_embeddings_mean = None
        self.class_embeddings_cov = None
        self.class_embeddings_std = None
        
        # Meta-learner for ensemble
        self.meta_classifier = None
        
        # Results storage
        self.results_dir = os.path.join(config.RESULTS_PATH, 'anomaly_detection_v3')
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("AnomalyDetectionProV3 initialized (CALIBRATED VERSION)")
    
    @torch.no_grad()
    def extract_embeddings_and_predictions(self):
        """Extract embeddings and predictions for all samples."""
        
        logger.info("="*80)
        logger.info("EXTRACTING EMBEDDINGS AND PREDICTIONS")
        logger.info("="*80)
        
        self.model.eval()
        
        embeddings = self.model.get_embeddings(
            self.graph_data.x,
            self.graph_data.edge_index,
            self.graph_data.edge_attr.squeeze()
        ).cpu().numpy()
        
        logits = self.model(
            self.graph_data.x,
            self.graph_data.edge_index,
            self.graph_data.edge_attr.squeeze()
        ).cpu().numpy()
        
        probabilities = F.softmax(torch.tensor(logits), dim=1).numpy()
        predictions = probabilities.argmax(axis=1)
        
        logger.info(f"Shape: {embeddings.shape}")
        logger.info("✓ Complete")
        
        return embeddings, logits, probabilities, predictions
    
    def compute_class_statistics_robust(self, embeddings, labels, use_indices=None):
        """
        Compute class statistics using train+val for better generalization.
        
        This reduces overfitting by using more diverse data for class centers.
        """
        
        logger.info("="*80)
        logger.info("COMPUTING CLASS STATISTICS (TRAIN+VAL)")
        logger.info("="*80)
        
        if use_indices is not None:
            embeddings = embeddings[use_indices]
            labels = labels[use_indices]
        
        n_classes = len(np.unique(labels))
        embedding_dim = embeddings.shape[1]
        
        self.class_embeddings_mean = np.zeros((n_classes, embedding_dim))
        self.class_embeddings_cov = []
        self.class_embeddings_std = np.zeros((n_classes, embedding_dim))
        
        for class_idx in range(n_classes):
            class_mask = labels == class_idx
            class_embeddings = embeddings[class_mask]
            
            if len(class_embeddings) < 5:
                logger.warning(f"Class {class_idx}: Only {len(class_embeddings)} samples")
                self.class_embeddings_mean[class_idx] = np.zeros(embedding_dim)
                self.class_embeddings_cov.append(np.eye(embedding_dim))
                self.class_embeddings_std[class_idx] = np.ones(embedding_dim)
                continue
            
            # Robust mean (using median for outlier resistance)
            mean = np.median(class_embeddings, axis=0)
            self.class_embeddings_mean[class_idx] = mean
            
            # Robust std (using MAD - Median Absolute Deviation)
            mad = np.median(np.abs(class_embeddings - mean), axis=0)
            std = 1.4826 * mad  # Convert MAD to std estimate
            std[std < 1e-6] = 1e-6  # Prevent division by zero
            self.class_embeddings_std[class_idx] = std
            
            # Robust covariance
            try:
                lw = LedoitWolf()
                lw.fit(class_embeddings)
                cov = lw.covariance_
            except:
                cov = np.cov(class_embeddings, rowvar=False)
                cov += np.eye(embedding_dim) * 1e-3
            
            self.class_embeddings_cov.append(cov)
            
            logger.info(f"Class {class_idx}: {len(class_embeddings)} samples")
        
        logger.info("✓ Complete (robust statistics)")
    
    def compute_anomaly_scores(self, embeddings, probabilities, predictions):
        """
        Compute 6 complementary anomaly scores (continuous, not binary).
        
        Returns scores in [0, 1] where higher = more anomalous.
        """
        
        logger.info("="*80)
        logger.info("COMPUTING ANOMALY SCORES (6 METHODS)")
        logger.info("="*80)
        
        n_samples = len(embeddings)
        scores = {}
        
        # ============================================================
        # METHOD 1: Confidence Score (1 - max_prob)
        # ============================================================
        confidence = probabilities.max(axis=1)
        scores['confidence'] = 1.0 - confidence
        logger.info(f"1. Confidence: range [{scores['confidence'].min():.3f}, {scores['confidence'].max():.3f}]")
        
        # ============================================================
        # METHOD 2: Prediction Margin (1 - margin)
        # ============================================================
        sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        scores['margin'] = 1.0 - margin
        logger.info(f"2. Margin: range [{scores['margin'].min():.3f}, {scores['margin'].max():.3f}]")
        
        # ============================================================
        # METHOD 3: Normalized Distance to Class Center
        # ============================================================
        if self.class_embeddings_mean is not None:
            distances = np.zeros(n_samples)
            
            for i in range(n_samples):
                pred_class = predictions[i]
                
                if pred_class >= len(self.class_embeddings_mean):
                    distances[i] = np.inf
                    continue
                
                # Normalized distance
                diff = embeddings[i] - self.class_embeddings_mean[pred_class]
                normalized_diff = diff / self.class_embeddings_std[pred_class]
                distances[i] = np.linalg.norm(normalized_diff)
            
            # Normalize to [0, 1] using robust percentiles
            valid_dist = distances[np.isfinite(distances)]
            if len(valid_dist) > 0:
                p5, p95 = np.percentile(valid_dist, [5, 95])
                scores['distance'] = np.clip((distances - p5) / (p95 - p5 + 1e-6), 0, 1)
            else:
                scores['distance'] = np.zeros(n_samples)
        else:
            scores['distance'] = np.zeros(n_samples)
        
        logger.info(f"3. Distance: range [{scores['distance'].min():.3f}, {scores['distance'].max():.3f}]")
        
        # ============================================================
        # METHOD 4: Isolation Forest Anomaly Score
        # ============================================================
        iso_forest = IsolationForest(
            contamination=0.1,
            random_state=self.config.RANDOM_SEED,
            n_estimators=100
        )
        iso_scores_raw = iso_forest.fit(embeddings).score_samples(embeddings)
        
        # Convert to [0, 1] (lower isolation score = more anomalous)
        scores['isolation'] = 1.0 / (1.0 + np.exp(iso_scores_raw))
        logger.info(f"4. Isolation: range [{scores['isolation'].min():.3f}, {scores['isolation'].max():.3f}]")
        
        # ============================================================
        # METHOD 5: Local Outlier Factor
        # ============================================================
        lof = LocalOutlierFactor(n_neighbors=20, novelty=False, contamination=0.1)
        lof.fit(embeddings)
        lof_scores_raw = lof.negative_outlier_factor_
        
        # Convert to [0, 1]
        scores['lof'] = 1.0 / (1.0 + np.exp(-(-lof_scores_raw - 1.0)))
        logger.info(f"5. LOF: range [{scores['lof'].min():.3f}, {scores['lof'].max():.3f}]")
        
        # ============================================================
        # METHOD 6: Prediction Entropy
        # ============================================================
        epsilon = 1e-10
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
        max_entropy = np.log(probabilities.shape[1])
        scores['entropy'] = entropy / max_entropy
        logger.info(f"6. Entropy: range [{scores['entropy'].min():.3f}, {scores['entropy'].max():.3f}]")
        
        logger.info("✓ Complete")
        
        return scores
    
    def calibrate_ensemble(self, val_scores, val_labels, val_predictions):
        """
        Calibrate ensemble on validation set to optimize error detection.
        
        Uses logistic regression meta-learner to find optimal weights.
        Target: maximize error detection while controlling false positives.
        """
        
        logger.info("="*80)
        logger.info("CALIBRATING ENSEMBLE ON VALIDATION SET")
        logger.info("="*80)
        
        # Create feature matrix from scores
        X_val = np.column_stack([
            val_scores['confidence'],
            val_scores['margin'],
            val_scores['distance'],
            val_scores['isolation'],
            val_scores['lof'],
            val_scores['entropy']
        ])
        
        # Target: 1 if prediction is wrong, 0 if correct
        y_val = (val_predictions != val_labels).astype(int)
        
        n_errors = y_val.sum()
        logger.info(f"Validation errors: {n_errors}/{len(y_val)} ({100*n_errors/len(y_val):.1f}%)")
        
        # Train meta-classifier with class balancing
        # This learns optimal combination of scores to detect errors
        self.meta_classifier = LogisticRegression(
            class_weight='balanced',  # Handle imbalanced classes
            random_state=self.config.RANDOM_SEED,
            max_iter=1000,
            penalty='l2',
            C=1.0
        )
        
        self.meta_classifier.fit(X_val, y_val)
        
        # Get learned weights
        self.calibrated_weights = self.meta_classifier.coef_[0]
        
        logger.info("\nLearned weights:")
        method_names = ['confidence', 'margin', 'distance', 'isolation', 'lof', 'entropy']
        for name, weight in zip(method_names, self.calibrated_weights):
            logger.info(f"  {name}: {weight:.3f}")
        
        # Calibrate threshold on validation set
        # Find threshold that gives ~5-7% anomaly rate while maximizing error detection
        val_anomaly_probs = self.meta_classifier.predict_proba(X_val)[:, 1]
        
        # Try different thresholds
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in np.linspace(0.1, 0.9, 50):
            val_anomalies = val_anomaly_probs >= threshold
            
            # Metrics
            tp = (val_anomalies & (y_val == 1)).sum()
            fp = (val_anomalies & (y_val == 0)).sum()
            fn = (~val_anomalies & (y_val == 1)).sum()
            
            if tp + fp > 0 and tp + fn > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    
                    # Prefer thresholds that give 5-10% anomaly rate
                    anomaly_rate = val_anomalies.sum() / len(val_anomalies)
                    
                    if 0.05 <= anomaly_rate <= 0.10:
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
        
        self.calibrated_threshold = best_threshold
        
        # Evaluate on validation
        val_final_anomalies = val_anomaly_probs >= self.calibrated_threshold
        
        tp = (val_final_anomalies & (y_val == 1)).sum()
        fp = (val_final_anomalies & (y_val == 0)).sum()
        fn = (~val_final_anomalies & (y_val == 1)).sum()
        tn = (~val_final_anomalies & (y_val == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"\nCalibrated threshold: {self.calibrated_threshold:.3f}")
        logger.info(f"Validation performance:")
        logger.info(f"  Anomaly rate: {100*val_final_anomalies.sum()/len(y_val):.1f}%")
        logger.info(f"  Error detection: {tp}/{n_errors} ({100*recall:.1f}%)")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1: {f1:.3f}")
        
        logger.info("✓ Calibration complete")
        
        return val_anomaly_probs
    
    def predict_anomalies(self, scores):
        """Use calibrated meta-classifier to predict anomalies."""
        
        X = np.column_stack([
            scores['confidence'],
            scores['margin'],
            scores['distance'],
            scores['isolation'],
            scores['lof'],
            scores['entropy']
        ])
        
        # Get probability of being an error
        anomaly_probs = self.meta_classifier.predict_proba(X)[:, 1]
        
        # Apply calibrated threshold
        is_anomaly = anomaly_probs >= self.calibrated_threshold
        
        return anomaly_probs, is_anomaly
    
    def analyze_split(self, split='test', embeddings_all=None, probabilities_all=None, 
                     predictions_all=None, labels_all=None, split_indices=None):
        """Analyze specific data split."""
        
        logger.info("\n" + "="*80)
        logger.info(f"ANALYZING {split.upper()} SET")
        logger.info("="*80)
        
        # Get indices
        if split == 'train':
            indices = split_indices['train_idx']
        elif split == 'val':
            indices = split_indices['val_idx']
        elif split == 'test':
            indices = split_indices['test_idx']
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Filter data
        embeddings = embeddings_all[indices]
        probabilities = probabilities_all[indices]
        predictions = predictions_all[indices]
        labels = labels_all[indices]
        
        logger.info(f"Samples: {len(indices)}")
        
        # Compute scores
        scores = self.compute_anomaly_scores(embeddings, probabilities, predictions)
        
        # For validation: calibrate ensemble
        if split == 'val':
            anomaly_probs = self.calibrate_ensemble(scores, labels, predictions)
            is_anomaly = anomaly_probs >= self.calibrated_threshold
        else:
            # For train/test: use calibrated model
            anomaly_probs, is_anomaly = self.predict_anomalies(scores)
        
        # Results
        correct = predictions == labels
        n_errors = (~correct).sum()
        errors_caught = (is_anomaly & ~correct).sum()
        
        results = {
            'split': split,
            'n_samples': len(indices),
            'indices': indices,
            'labels': labels,
            'predictions': predictions,
            'probabilities': probabilities,
            'embeddings': embeddings,
            'scores': scores,
            'anomaly_probs': anomaly_probs,
            'is_anomaly': is_anomaly,
            'n_errors': int(n_errors),
            'errors_caught': int(errors_caught),
            'error_catch_rate': float(100 * errors_caught / n_errors) if n_errors > 0 else 0.0
        }
        
        logger.info("\nPERFORMANCE:")
        logger.info(f"  Anomalies flagged: {is_anomaly.sum()} ({100*is_anomaly.sum()/len(indices):.1f}%)")
        logger.info(f"  Errors in dataset: {n_errors} ({100*n_errors/len(indices):.1f}%)")
        logger.info(f"  Errors caught: {errors_caught}/{n_errors} ({results['error_catch_rate']:.1f}%)")
        
        logger.info("✓ Complete")
        
        return results
    
    def visualize_results(self, results, save_dir):
        """Create comprehensive visualization."""
        
        logger.info(f"Creating visualization for {results['split']}...")
        
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        is_anomaly = results['is_anomaly']
        anomaly_probs = results['anomaly_probs']
        correct = results['predictions'] == results['labels']
        
        # Panel 1: Anomaly probability distribution
        ax1 = fig.add_subplot(gs[0, 0])
        bins = np.linspace(0, 1, 50)
        ax1.hist(anomaly_probs[~is_anomaly], bins=bins, alpha=0.7, label='Normal',
                color='green', edgecolor='black')
        ax1.hist(anomaly_probs[is_anomaly], bins=bins, alpha=0.7, label='Flagged',
                color='red', edgecolor='black')
        ax1.axvline(self.calibrated_threshold, color='orange', linestyle='--', 
                   linewidth=3, label=f'Threshold ({self.calibrated_threshold:.2f})')
        ax1.set_xlabel('Anomaly Probability', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Calibrated Anomaly Scores', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
        
        # Panel 2: Method contributions (weights)
        ax2 = fig.add_subplot(gs[0, 1])
        methods = ['Confidence', 'Margin', 'Distance', 'Isolation', 'LOF', 'Entropy']
        weights = self.calibrated_weights
        colors_weights = ['#e74c3c' if w > 0 else '#3498db' for w in weights]
        bars = ax2.barh(methods, weights, color=colors_weights, edgecolor='black', linewidth=2)
        ax2.set_xlabel('Learned Weight', fontsize=12, fontweight='bold')
        ax2.set_title('Meta-Learned Method Importance', fontsize=14, fontweight='bold')
        ax2.axvline(0, color='black', linewidth=2)
        ax2.grid(axis='x', alpha=0.3)
        for bar, weight in zip(bars, weights):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, f'{weight:.3f}',
                    ha='left' if width > 0 else 'right', va='center',
                    fontsize=11, fontweight='bold')
        
        # Panel 3: Error detection breakdown
        ax3 = fig.add_subplot(gs[0, 2])
        categories = ['Errors\nCaught', 'Errors\nMissed', 'False\nAlarms']
        counts = [
            (is_anomaly & ~correct).sum(),
            (~is_anomaly & ~correct).sum(),
            (is_anomaly & correct).sum()
        ]
        colors_cat = ['#27ae60', '#e74c3c', '#f39c12']
        bars = ax3.bar(categories, counts, color=colors_cat, edgecolor='black', linewidth=2)
        ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax3.set_title('Error Detection Breakdown', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(count)}', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
        
        # Panel 4: PCA embedding
        ax4 = fig.add_subplot(gs[1, :])
        pca = PCA(n_components=2, random_state=self.config.RANDOM_SEED)
        embeddings_2d = pca.fit_transform(results['embeddings'])
        
        # Plot by correctness and anomaly status
        mask_correct_normal = correct & ~is_anomaly
        mask_correct_anomaly = correct & is_anomaly
        mask_error_normal = ~correct & ~is_anomaly
        mask_error_anomaly = ~correct & is_anomaly
        
        ax4.scatter(embeddings_2d[mask_correct_normal, 0], embeddings_2d[mask_correct_normal, 1],
                   c='lightblue', s=20, alpha=0.5, label='Correct (Normal)', edgecolors='none')
        ax4.scatter(embeddings_2d[mask_correct_anomaly, 0], embeddings_2d[mask_correct_anomaly, 1],
                   c='orange', s=80, alpha=0.7, label='Correct (Flagged)', 
                   edgecolors='darkorange', linewidth=1.5, marker='o')
        ax4.scatter(embeddings_2d[mask_error_normal, 0], embeddings_2d[mask_error_normal, 1],
                   c='pink', s=80, alpha=0.7, label='Error (Missed)', 
                   edgecolors='red', linewidth=1.5, marker='s')
        ax4.scatter(embeddings_2d[mask_error_anomaly, 0], embeddings_2d[mask_error_anomaly, 1],
                   c='red', s=120, alpha=0.9, label='Error (Caught)', 
                   edgecolors='darkred', linewidth=2, marker='X')
        
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                      fontsize=12, fontweight='bold')
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', 
                      fontsize=12, fontweight='bold')
        ax4.set_title('Embedding Space with Error Detection', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11, loc='best')
        ax4.grid(alpha=0.3)
        
        # Panel 5: Score distributions (6 methods)
        ax5 = fig.add_subplot(gs[2, :2])
        
        method_names = ['Confidence', 'Margin', 'Distance', 'Isolation', 'LOF', 'Entropy']
        score_keys = ['confidence', 'margin', 'distance', 'isolation', 'lof', 'entropy']
        
        positions = np.arange(len(method_names))
        width = 0.35
        
        means_normal = [results['scores'][k][~is_anomaly].mean() for k in score_keys]
        means_anomaly = [results['scores'][k][is_anomaly].mean() for k in score_keys]
        
        bars1 = ax5.bar(positions - width/2, means_normal, width, 
                       label='Normal', color='green', alpha=0.7, edgecolor='black')
        bars2 = ax5.bar(positions + width/2, means_anomaly, width,
                       label='Anomaly', color='red', alpha=0.7, edgecolor='black')
        
        ax5.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
        ax5.set_title('Method Score Comparison', fontsize=14, fontweight='bold')
        ax5.set_xticks(positions)
        ax5.set_xticklabels(method_names, rotation=45, ha='right')
        ax5.legend(fontsize=11)
        ax5.grid(axis='y', alpha=0.3)
        
        # Panel 6: Performance metrics
        ax6 = fig.add_subplot(gs[2, 2])
        
        n_total = len(results['labels'])
        n_errors = results['n_errors']
        errors_caught = results['errors_caught']
        false_positives = (is_anomaly & correct).sum()
        
        metrics = {
            'Error\nDetection': 100 * errors_caught / n_errors if n_errors > 0 else 0,
            'False\nPositive': 100 * false_positives / n_total,
            'Anomaly\nRate': 100 * is_anomaly.sum() / n_total
        }
        
        bars = ax6.bar(metrics.keys(), metrics.values(), 
                      color=['#27ae60', '#e74c3c', '#3498db'],
                      edgecolor='black', linewidth=2)
        ax6.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax6.set_title('Key Metrics', fontsize=14, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        ax6.set_ylim([0, 100])
        
        for bar, (name, value) in zip(bars, metrics.items()):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2, height,
                    f'{value:.1f}%', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Anomaly Detection V3 - {results["split"].upper()} Set (Calibrated)',
                    fontsize=18, fontweight='bold', y=0.995)
        
        save_path = os.path.join(save_dir, f'anomaly_v3_{results["split"]}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved: {save_path}")
    
    def generate_report(self, results, save_dir):
        """Generate detailed CSV report."""
        
        logger.info("Generating report...")
        
        from backend.utils.data_processor import DataProcessor
        processor = DataProcessor(self.config)
        processed_data = processor.load_processed_data(self.config.PROCESSED_DATA_PATH)
        cancer_types = processed_data['cancer_types']
        sample_ids = processed_data['sample_ids']
        
        anomaly_indices = np.where(results['is_anomaly'])[0]
        
        report_data = []
        
        for idx in anomaly_indices:
            global_idx = results['indices'][idx]
            true_label = results['labels'][idx]
            pred_label = results['predictions'][idx]
            
            report_data.append({
                'sample_id': sample_ids[global_idx],
                'true_cancer': cancer_types[true_label],
                'predicted_cancer': cancer_types[pred_label],
                'is_correct': true_label == pred_label,
                'anomaly_probability': results['anomaly_probs'][idx],
                'confidence_score': results['scores']['confidence'][idx],
                'margin_score': results['scores']['margin'][idx],
                'distance_score': results['scores']['distance'][idx],
                'isolation_score': results['scores']['isolation'][idx],
                'lof_score': results['scores']['lof'][idx],
                'entropy_score': results['scores']['entropy'][idx]
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('anomaly_probability', ascending=False)
        
        report_path = os.path.join(save_dir, f'anomaly_report_v3_{results["split"]}.csv')
        report_df.to_csv(report_path, index=False)
        
        logger.info(f"Report saved: {report_path}")
        
        # Summary
        summary = {
            'split': results['split'],
            'n_samples': results['n_samples'],
            'n_anomalies': int(results['is_anomaly'].sum()),
            'pct_anomalies': float(100 * results['is_anomaly'].sum() / results['n_samples']),
            'n_errors': results['n_errors'],
            'errors_caught': results['errors_caught'],
            'error_catch_rate': results['error_catch_rate'],
            'calibrated_threshold': float(self.calibrated_threshold),
            'learned_weights': {
                'confidence': float(self.calibrated_weights[0]),
                'margin': float(self.calibrated_weights[1]),
                'distance': float(self.calibrated_weights[2]),
                'isolation': float(self.calibrated_weights[3]),
                'lof': float(self.calibrated_weights[4]),
                'entropy': float(self.calibrated_weights[5])
            }
        }
        
        summary_path = os.path.join(save_dir, f'anomaly_summary_v3_{results["split"]}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Summary saved: {summary_path}")
        
        return report_df, summary
    
    def run_complete_analysis(self):
        """
        Run complete calibrated analysis on all splits.
        
        Workflow:
        1. Extract all data
        2. Compute class statistics on train+val (reduce overfitting)
        3. Calibrate on validation set
        4. Evaluate on train/val/test
        """
        
        logger.info("\n" + "="*80)
        logger.info("ANOMALY DETECTION V3 - CALIBRATED ANALYSIS")
        logger.info("="*80 + "\n")
        
        # Load data
        from backend.utils.data_processor import DataProcessor
        processor = DataProcessor(self.config)
        processed_data = processor.load_processed_data(self.config.PROCESSED_DATA_PATH)
        
        split_indices = processed_data['split_indices']
        
        # Extract all embeddings and predictions
        embeddings_all, logits_all, probabilities_all, predictions_all = \
            self.extract_embeddings_and_predictions()
        
        labels_all = self.graph_data.y.cpu().numpy()
        
        # Compute class statistics on train+val (better generalization)
        train_val_indices = np.concatenate([
            split_indices['train_idx'],
            split_indices['val_idx']
        ])
        
        self.compute_class_statistics_robust(
            embeddings_all, 
            labels_all, 
            use_indices=train_val_indices
        )
        
        # Analyze all splits
        all_results = []
        all_summaries = []
        
        for split in ['val', 'train', 'test']:  # Val first for calibration
            logger.info(f"\n{'='*80}")
            logger.info(f"{split.upper()} SET")
            logger.info('='*80)
            
            results = self.analyze_split(
                split=split,
                embeddings_all=embeddings_all,
                probabilities_all=probabilities_all,
                predictions_all=predictions_all,
                labels_all=labels_all,
                split_indices=split_indices
            )
            
            self.visualize_results(results, self.results_dir)
            report_df, summary = self.generate_report(results, self.results_dir)
            
            all_results.append(results)
            all_summaries.append(summary)
            
            logger.info(f"✓ {split.upper()} complete")
        
        # Combined summary
        combined = pd.DataFrame(all_summaries)
        combined_path = os.path.join(self.results_dir, 'anomaly_v3_summary_all.csv')
        combined.to_csv(combined_path, index=False)
        
        logger.info("\n" + "="*80)
        logger.info("✓✓✓ ANOMALY DETECTION V3 COMPLETE ✓✓✓")
        logger.info("="*80)
        logger.info(f"Results: {self.results_dir}")
        logger.info("\nFINAL PERFORMANCE:")
        for summary in all_summaries:
            logger.info(f"{summary['split'].upper()}: "
                       f"Caught {summary['errors_caught']}/{summary['n_errors']} errors "
                       f"({summary['error_catch_rate']:.1f}%), "
                       f"Flagged {summary['n_anomalies']} samples ({summary['pct_anomalies']:.1f}%)")
        logger.info("="*80 + "\n")
        
        return combined