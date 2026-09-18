import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    silhouette_samples
)
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import kruskal
import logging
import os
from typing import Dict, Tuple, List
import json
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')


class BiologyInformedSubtypeDiscovery:
    """
    Biology-first molecular subtype discovery.
    Uses domain knowledge about expected cancer heterogeneity
    combined with statistical validation.
    """
    
    def __init__(self, model, graph_data, config, device='cpu'):
        self.model = model
        self.graph_data = graph_data.to(device)
        self.config = config
        self.device = device
        
        # Gene information
        self.gene_names = None
        self.load_gene_information()
        
        # Expected subtype ranges based on cancer biology
        self.expected_subtypes = {
            'BRCA': (4, 8),    # Luminal A/B, HER2+, Basal, Normal-like + finer subtypes
            'LUAD': (3, 6),    # Terminal respiratory unit, proximal inflammatory, proximal proliferative
            'LUSC': (3, 5),    # Classical, basal, secretory, primitive
            'COAD': (4, 6),    # CMS1-4 + additional subtypes
            'KIRC': (3, 5),    # ccA, ccB + additional
            'PRAD': (3, 6),    # ERG+, ETS+, SPOP, etc.
            'GBM': (3, 5),     # Proneural, Neural, Classical, Mesenchymal
            'OV': (4, 6),      # Differentiated, immunoreactive, mesenchymal, proliferative
            'HNSC': (3, 5),    # HPV+, basal, mesenchymal, atypical
            'THCA': (2, 4),    # Classical, follicular variant
            'default': (3, 8)  # For other cancers
        }
        
        # Results storage
        self.results_dir = os.path.join(config.RESULTS_PATH, 'biology_informed_subtypes')
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("BiologyInformedSubtypeDiscovery initialized")
    
    def load_gene_information(self):
        """Load gene names from processed data."""
        try:
            from backend.utils.data_processor import DataProcessor
            processor = DataProcessor(self.config)
            processed_data = processor.load_processed_data(self.config.PROCESSED_DATA_PATH)
            self.gene_names = processed_data['gene_names']
            logger.info(f"Loaded {len(self.gene_names)} gene names")
        except Exception as e:
            logger.warning(f"Could not load gene names: {e}")
            self.gene_names = [f"Gene_{i}" for i in range(self.graph_data.num_node_features)]
    
    @torch.no_grad()
    def extract_embeddings(self, cancer_type_idx):
        """Extract embeddings for specific cancer type."""
        
        self.model.eval()
        
        # Get all embeddings
        embeddings = self.model.get_embeddings(
            self.graph_data.x,
            self.graph_data.edge_index,
            self.graph_data.edge_attr.squeeze()
        ).cpu().numpy()
        
        labels = self.graph_data.y.cpu().numpy()
        
        # Filter to cancer type
        mask = labels == cancer_type_idx
        cancer_embeddings = embeddings[mask]
        cancer_features = self.graph_data.x[mask].cpu().numpy()
        
        return cancer_embeddings, cancer_features
    
    def reduce_dimensions(self, embeddings):
        """Apply UMAP for dimensionality reduction."""
        
        # PCA preprocessing if needed
        if embeddings.shape[1] > 50:
            n_comp = min(50, embeddings.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=self.config.RANDOM_SEED)
            embeddings = pca.fit_transform(embeddings)
        
        # UMAP
        n_neighbors = min(30, embeddings.shape[0] - 1)  # Increased from 15
        umap_reducer = UMAP(
            n_components=2, 
            n_neighbors=n_neighbors,
            min_dist=0.0,  # Allow tighter clusters
            metric='cosine',
            random_state=self.config.RANDOM_SEED,
            verbose=False
        )
        
        reduced = umap_reducer.fit_transform(embeddings)
        
        return reduced
    
    def find_stable_clusters_hierarchical(self, data, min_k, max_k):
        """
        Use hierarchical clustering with silhouette analysis
        to find stable clusters within expected range.
        """
        
        logger.info(f"Hierarchical clustering analysis (k={min_k} to {max_k})...")
        
        # Hierarchical clustering
        linkage_matrix = linkage(data, method='ward')
        
        results = {}
        
        for k in range(min_k, max_k + 1):
            labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1
            
            # Compute metrics
            sil = silhouette_score(data, labels)
            db = davies_bouldin_score(data, labels)
            ch = calinski_harabasz_score(data, labels)
            
            # Check cluster balance
            unique, counts = np.unique(labels, return_counts=True)
            min_size = counts.min()
            max_size = counts.max()
            balance_ratio = min_size / max_size
            
            results[k] = {
                'labels': labels,
                'silhouette': sil,
                'davies_bouldin': db,
                'calinski_harabasz': ch,
                'balance_ratio': balance_ratio,
                'min_cluster_size': min_size
            }
            
            logger.info(f"  k={k}: Sil={sil:.3f}, DB={db:.3f}, Balance={balance_ratio:.2f}")
        
        return results, linkage_matrix
    
    def find_stable_clusters_gmm(self, data, min_k, max_k):
        """
        Use Gaussian Mixture Models with BIC/AIC
        to find natural number of components.
        """
        
        logger.info(f"GMM analysis (k={min_k} to {max_k})...")
        
        results = {}
        
        for k in range(min_k, max_k + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                random_state=self.config.RANDOM_SEED,
                n_init=10
            )
            
            labels = gmm.fit_predict(data)
            
            bic = gmm.bic(data)
            aic = gmm.aic(data)
            
            # Silhouette
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(data, labels)
            else:
                sil = 0.0
            
            results[k] = {
                'labels': labels,
                'bic': bic,
                'aic': aic,
                'silhouette': sil
            }
            
            logger.info(f"  k={k}: BIC={bic:.1f}, AIC={aic:.1f}, Sil={sil:.3f}")
        
        return results
    
    def compute_cluster_separability(self, data, labels):
        """
        Compute how well-separated clusters are using multiple metrics.
        """
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return 0.0
        
        # Silhouette per sample
        sil_samples = silhouette_samples(data, labels)
        
        # Mean silhouette per cluster
        cluster_sils = []
        for label in unique_labels:
            cluster_sil = sil_samples[labels == label].mean()
            cluster_sils.append(cluster_sil)
        
        # Minimum cluster silhouette (weakest cluster)
        min_cluster_sil = min(cluster_sils)
        
        # Overall silhouette
        overall_sil = sil_samples.mean()
        
        # Negative silhouette samples (potential misclassifications)
        n_negative = (sil_samples < 0).sum()
        negative_ratio = n_negative / len(sil_samples)
        
        return {
            'overall_silhouette': overall_sil,
            'min_cluster_silhouette': min_cluster_sil,
            'negative_ratio': negative_ratio,
            'cluster_silhouettes': cluster_sils
        }
    
    def biological_validation(self, features, labels, top_n=20):
        """Find differentially expressed genes."""
        
        unique_clusters = np.unique(labels)
        n_clusters = len(unique_clusters)
        
        if n_clusters < 2:
            return None
        
        p_values = []
        
        for gene_idx in range(min(features.shape[1], 5000)):  # Limit for speed
            gene_expression = features[:, gene_idx]
            groups = [gene_expression[labels == c] for c in unique_clusters]
            
            try:
                stat, p_val = kruskal(*groups)
                p_values.append((gene_idx, p_val))
            except:
                p_values.append((gene_idx, 1.0))
        
        # Sort by p-value
        p_values.sort(key=lambda x: x[1])
        
        de_genes = []
        for gene_idx, p_val in p_values[:top_n]:
            de_genes.append({
                'gene': self.gene_names[gene_idx],
                'p_value': float(p_val),
                'gene_index': int(gene_idx)
            })
        
        return de_genes
    
    def select_optimal_k_biology_informed(self, hierarchical_results, gmm_results, 
                                         cancer_name, n_samples):
        """
        Select optimal k using biology-informed criteria.
        """
        
        logger.info("\n" + "="*60)
        logger.info("Biology-Informed Optimal K Selection")
        logger.info("="*60)
        
        # Get expected range
        if cancer_name in self.expected_subtypes:
            min_expected, max_expected = self.expected_subtypes[cancer_name]
        else:
            min_expected, max_expected = self.expected_subtypes['default']
        
        # Adjust for sample size
        if n_samples < 100:
            max_expected = min(max_expected, 4)
        elif n_samples < 200:
            max_expected = min(max_expected, 6)
        
        logger.info(f"Expected range for {cancer_name}: {min_expected}-{max_expected} subtypes")
        logger.info(f"Sample size: {n_samples}")
        
        # Score each k
        k_values = sorted(hierarchical_results.keys())
        scores = {}
        
        for k in k_values:
            h_res = hierarchical_results[k]
            g_res = gmm_results[k]
            
            score = 0.0
            
            # 1. Biology prior: strongly favor expected range
            if min_expected <= k <= max_expected:
                score += 3.0  # Strong bonus for being in expected range
            elif k < min_expected:
                score += 1.0 - (min_expected - k) * 0.5  # Penalty for too few
            else:
                score += 1.0 - (k - max_expected) * 0.3  # Smaller penalty for too many
            
            # 2. Statistical quality (normalized)
            score += h_res['silhouette'] * 1.0
            score += (1.0 / (1.0 + h_res['davies_bouldin'])) * 0.5
            score += h_res['balance_ratio'] * 0.5
            
            # 3. Minimum cluster size penalty
            if h_res['min_cluster_size'] < 10:
                score -= 1.0
            elif h_res['min_cluster_size'] < 20:
                score -= 0.5
            
            # 4. GMM support (lower BIC = better)
            bic_norm = 1.0 - (g_res['bic'] - min([gmm_results[ki]['bic'] for ki in k_values])) / \
                       (max([gmm_results[ki]['bic'] for ki in k_values]) - min([gmm_results[ki]['bic'] for ki in k_values]) + 1e-10)
            score += bic_norm * 0.5
            
            scores[k] = score
            
            logger.info(f"k={k}: Score={score:.3f} "
                       f"(Sil={h_res['silhouette']:.3f}, "
                       f"DB={h_res['davies_bouldin']:.3f}, "
                       f"MinSize={h_res['min_cluster_size']}, "
                       f"Balance={h_res['balance_ratio']:.2f})")
        
        # Select best k
        optimal_k = max(scores, key=scores.get)
        
        logger.info(f"\n✓ Selected k={optimal_k} (Score={scores[optimal_k]:.3f})")
        
        return optimal_k
    
    def visualize_subtypes(self, cancer_name, data, labels, de_genes, 
                          metrics, n_samples, save_dir):
        """Create comprehensive visualization."""
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        n_subtypes = len(np.unique(labels))
        
        # Plot 1: UMAP with clusters
        ax1 = fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter(data[:, 0], data[:, 1], 
                            c=labels, cmap='tab20', s=60, alpha=0.7, 
                            edgecolors='k', linewidth=0.2)
        ax1.set_xlabel('UMAP 1', fontsize=11)
        ax1.set_ylabel('UMAP 2', fontsize=11)
        ax1.set_title(f'{cancer_name} Molecular Subtypes\n{n_subtypes} Subtypes Identified', 
                     fontsize=13, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Subtype', fontsize=10)
        
        # Plot 2: Subtype sizes
        ax2 = fig.add_subplot(gs[0, 1])
        unique_labels, counts = np.unique(labels, return_counts=True)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        bars = ax2.bar(unique_labels, counts, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Subtype ID', fontsize=11)
        ax2.set_ylabel('Number of Patients', fontsize=11)
        ax2.set_title('Subtype Distribution', fontsize=13, fontweight='bold')
        ax2.set_xticks(unique_labels)
        
        # Add percentage labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pct = 100 * count / n_samples
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Quality metrics
        ax3 = fig.add_subplot(gs[0, 2])
        metric_names = ['Silhouette\n(0-1)', 'Davies-Bouldin\n(lower=better)', 
                       'Balance\nRatio']
        metric_values = [
            metrics.get('silhouette', 0),
            min(metrics.get('davies_bouldin', 1), 2.0),  # Cap at 2 for visualization
            metrics.get('balance_ratio', 0)
        ]
        colors_bars = ['#27ae60', '#e74c3c', '#3498db']
        bars = ax3.bar(range(len(metric_names)), metric_values, 
                      color=colors_bars, edgecolor='black', linewidth=1.5)
        ax3.set_xticks(range(len(metric_names)))
        ax3.set_xticklabels(metric_names, fontsize=9)
        ax3.set_ylabel('Score', fontsize=11)
        ax3.set_title('Clustering Quality', fontsize=13, fontweight='bold')
        ax3.set_ylim(0, max(metric_values) * 1.2)
        
        # Add value labels
        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Plot 4: Top DE genes
        if de_genes and len(de_genes) > 0:
            ax4 = fig.add_subplot(gs[1, :])
            top_genes = de_genes[:15]
            genes = [g['gene'] for g in top_genes]
            p_vals = [-np.log10(max(g['p_value'], 1e-300)) for g in top_genes]
            
            bars = ax4.barh(range(len(genes)), p_vals, color='steelblue', 
                          edgecolor='navy', linewidth=1.2)
            ax4.set_yticks(range(len(genes)))
            ax4.set_yticklabels(genes, fontsize=10, fontweight='bold')
            ax4.set_xlabel('-log10(p-value)', fontsize=11)
            ax4.set_title('Top Differentially Expressed Biomarker Genes', 
                         fontsize=13, fontweight='bold')
            ax4.axvline(x=-np.log10(0.05), color='red', linestyle='--', 
                       linewidth=2.5, label='p=0.05 threshold', alpha=0.8)
            ax4.axvline(x=-np.log10(0.001), color='darkred', linestyle='--', 
                       linewidth=2.5, label='p=0.001 threshold', alpha=0.8)
            ax4.legend(fontsize=10, loc='lower right')
            ax4.invert_yaxis()
            ax4.grid(axis='x', alpha=0.3)
        
        # Plot 5: Summary table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        table_data = []
        for i, (subtype, count) in enumerate(zip(unique_labels, counts)):
            percentage = 100 * count / n_samples
            table_data.append([
                f'Subtype {subtype}',
                f'{count}',
                f'{percentage:.1f}%',
                f'{metrics.get("cluster_silhouettes", [0]*n_subtypes)[i]:.3f}' if i < len(metrics.get("cluster_silhouettes", [])) else 'N/A'
            ])
        
        table = ax5.table(
            cellText=table_data,
            colLabels=['Subtype', 'N Patients', 'Percentage', 'Silhouette'],
            cellLoc='center',
            loc='center',
            colWidths=[0.25, 0.25, 0.25, 0.25]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style
        for i in range(4):
            table[(0, i)].set_facecolor('#2c3e50')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
        
        for i in range(1, len(table_data) + 1):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
                table[(i, j)].set_edgecolor('black')
                table[(i, j)].set_linewidth(1.5)
        
        plt.suptitle(f'Biology-Informed Molecular Subtype Analysis: {cancer_name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        save_path = os.path.join(save_dir, f'biology_subtypes_{cancer_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Visualization saved to {save_path}")
    
    def analyze_cancer_type(self, cancer_type_idx):
        """Complete biology-informed analysis for one cancer type."""
        
        # Get cancer name
        try:
            from backend.utils.data_processor import DataProcessor
            processor = DataProcessor(self.config)
            processed_data = processor.load_processed_data(self.config.PROCESSED_DATA_PATH)
            cancer_types = processed_data['cancer_types']
            cancer_name = cancer_types[cancer_type_idx]
        except:
            cancer_name = f'Cancer_{cancer_type_idx}'
        
        # Get data
        labels = self.graph_data.y.cpu().numpy()
        n_samples = np.sum(labels == cancer_type_idx)
        
        if n_samples < 20:
            logger.warning(f"Skipping {cancer_name}: only {n_samples} samples")
            return None
        
        logger.info("="*60)
        logger.info(f"Analyzing {cancer_name} ({n_samples} samples)")
        logger.info("="*60)
        
        # Extract data
        embeddings, features = self.extract_embeddings(cancer_type_idx)
        reduced = self.reduce_dimensions(embeddings)
        
        # Get expected range
        if cancer_name in self.expected_subtypes:
            min_k, max_k = self.expected_subtypes[cancer_name]
        else:
            min_k, max_k = self.expected_subtypes['default']
        
        # Adjust for sample size
        if n_samples < 50:
            min_k, max_k = 2, 4
        elif n_samples < 100:
            min_k, max_k = 2, min(max_k, 5)
        elif n_samples < 200:
            min_k, max_k = 3, min(max_k, 7)
        
        max_k = min(max_k, n_samples // 20)
        
        # Run clustering analyses
        h_results, linkage_mat = self.find_stable_clusters_hierarchical(reduced, min_k, max_k)
        g_results = self.find_stable_clusters_gmm(reduced, min_k, max_k)
        
        # Select optimal k
        optimal_k = self.select_optimal_k_biology_informed(
            h_results, g_results, cancer_name, n_samples
        )
        
        # Get final clustering
        final_labels = h_results[optimal_k]['labels']
        
        # Compute detailed separability
        separability = self.compute_cluster_separability(reduced, final_labels)
        
        # Biological validation
        de_genes = self.biological_validation(features, final_labels, top_n=20)
        
        # Combine metrics
        metrics = h_results[optimal_k].copy()
        metrics.update(separability)
        
        # Visualize
        self.visualize_subtypes(
            cancer_name, reduced, final_labels, de_genes,
            metrics, n_samples, self.results_dir
        )
        
        # Save results
        output = {
            'cancer_type': cancer_name,
            'n_samples': int(n_samples),
            'optimal_k': int(optimal_k),
            'expected_range': self.expected_subtypes.get(cancer_name, self.expected_subtypes['default']),
            'silhouette': float(metrics['silhouette']),
            'davies_bouldin': float(metrics['davies_bouldin']),
            'calinski_harabasz': float(metrics['calinski_harabasz']),
            'balance_ratio': float(metrics['balance_ratio']),
            'overall_silhouette': float(separability['overall_silhouette']),
            'min_cluster_silhouette': float(separability['min_cluster_silhouette']),
            'negative_ratio': float(separability['negative_ratio']),
            'de_genes': de_genes[:10] if de_genes else [],
            'cluster_sizes': [int(np.sum(final_labels == i)) for i in range(optimal_k)]
        }
        
        json_path = os.path.join(self.results_dir, f'results_{cancer_name}.json')
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=4)
        
        return output
    
    def analyze_all_cancer_types(self):
        """Analyze all 33 cancer types with biology-informed approach."""
        
        logger.info("="*60)
        logger.info("BIOLOGY-INFORMED MOLECULAR SUBTYPE DISCOVERY")
        logger.info("="*60)
        
        labels = self.graph_data.y.cpu().numpy()
        unique_labels = np.unique(labels)
        
        all_results = []
        
        for i, cancer_idx in enumerate(unique_labels):
            logger.info(f"\n[{i+1}/{len(unique_labels)}]")
            
            result = self.analyze_cancer_type(cancer_idx)
            
            if result:
                all_results.append(result)
        
        # Summary
        summary_df = pd.DataFrame(all_results)
        summary_df = summary_df.sort_values('n_samples', ascending=False)
        
        summary_path = os.path.join(self.results_dir, 'biology_informed_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        logger.info("="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Results: {self.results_dir}")
        
        return summary_df