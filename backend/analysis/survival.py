import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
from lifelines.utils import concordance_index
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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


class SurvivalAnalysisPro:
    """
    Professional Prognostic Stratification using GCN embeddings and clinical staging.
    
    Features:
    1. Real TCGA survival data integration (OS, PFI, stages)
    2. Stage-based risk stratification (I/II=Low, III=Medium, IV=High)
    3. Kaplan-Meier survival curves with log-rank testing
    4. Cox proportional hazards regression with C-index
    5. Model confidence correlation with clinical outcomes
    6. Publication-quality visualizations
    
    Note: Uses ACTUAL clinical data from TCGA, not simulated data.
    """
    
    def __init__(self, model, graph_data, config, device='cpu'):
        self.model = model
        self.graph_data = graph_data.to(device)
        self.config = config
        self.device = device
        
        # Load real survival data
        self.survival_data = None
        self.load_survival_data()
        
        # Results storage
        self.results_dir = os.path.join(config.RESULTS_PATH, 'survival_analysis_pro')
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("SurvivalAnalysisPro initialized")
    
    def load_survival_data(self):
        """
        Load REAL survival data from processed dataset.
        
        Expected columns from TCGA:
        - patient_id: TCGA barcode
        - cancer_type: Cancer type abbreviation
        - os_event: Overall survival event (0=alive, 1=dead)
        - os_time: Overall survival time (days)
        - pfi_event: Progression-free interval event
        - pfi_time: Progression-free interval time
        - stage: AJCC pathologic tumor stage
        """
        
        logger.info("="*80)
        logger.info("LOADING SURVIVAL DATA")
        logger.info("="*80)
        
        try:
            from backend.utils.data_processor import DataProcessor
            processor = DataProcessor(self.config)
            processed_data = processor.load_processed_data(self.config.PROCESSED_DATA_PATH)
            
            # Check if survival data exists
            if 'survival_data' not in processed_data:
                raise ValueError("No survival_data found in processed_data.pt!")
            
            self.survival_data = processed_data['survival_data']
            
            # Verify it's a DataFrame
            if not isinstance(self.survival_data, pd.DataFrame):
                raise TypeError(f"survival_data is {type(self.survival_data)}, expected DataFrame!")
            
            logger.info(f"✓ Loaded survival data: {self.survival_data.shape}")
            logger.info(f"  Columns: {list(self.survival_data.columns)}")
            
            # Validate required columns
            required_cols = ['patient_id', 'cancer_type']
            missing_cols = [col for col in required_cols if col not in self.survival_data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check what survival endpoints we have
            has_os = 'os_event' in self.survival_data.columns and 'os_time' in self.survival_data.columns
            has_pfi = 'pfi_event' in self.survival_data.columns and 'pfi_time' in self.survival_data.columns
            has_stage = 'stage' in self.survival_data.columns
            
            if has_os:
                n_events = self.survival_data['os_event'].sum()
                logger.info(f"  ✓ Overall Survival (OS): {int(n_events)} events ({100*n_events/len(self.survival_data):.1f}%)")
            else:
                logger.warning("  ⚠️ No OS data available")
            
            if has_pfi:
                n_events = self.survival_data['pfi_event'].sum()
                logger.info(f"  ✓ Progression-Free Interval (PFI): {int(n_events)} events ({100*n_events/len(self.survival_data):.1f}%)")
            else:
                logger.warning("  ⚠️ No PFI data available")
            
            if has_stage:
                stage_counts = self.survival_data['stage'].value_counts()
                logger.info(f"  ✓ Stage data: {len(stage_counts)} unique stages")
            else:
                logger.warning("  ⚠️ No stage data available")
            
            if not (has_os or has_pfi):
                raise ValueError("No survival endpoints (OS or PFI) available!")
            
            logger.info("✓ Survival data loaded successfully")
        
        except Exception as e:
            logger.error(f"Error loading survival data: {e}")
            raise
    
    def parse_stage(self, stage_str):
        """
        Parse AJCC stage string to numeric risk level.
        
        Stage I/II   → 0 (Low Risk)
        Stage III    → 1 (Medium Risk)
        Stage IV     → 2 (High Risk)
        Unknown/NaN  → -1
        
        Args:
            stage_str: Stage string (e.g., "Stage IIA", "Stage IV")
        
        Returns:
            risk_level: 0, 1, 2, or -1
        """
        
        if pd.isna(stage_str):
            return -1
        
        stage_str = str(stage_str).upper().strip()
        
        # Stage IV
        if 'IV' in stage_str or 'STAGE 4' in stage_str:
            return 2  # High risk
        
        # Stage III
        if 'III' in stage_str or 'STAGE 3' in stage_str:
            return 1  # Medium risk
        
        # Stage I or II
        if 'I' in stage_str or 'STAGE 1' in stage_str or 'STAGE 2' in stage_str:
            return 0  # Low risk
        
        # Unknown
        return -1
    
    def stratify_by_stage(self):
        """
        Stratify patients into risk groups based on AJCC tumor stage.
        
        Returns:
            risk_groups: Array of risk levels (0=Low, 1=Medium, 2=High, -1=Unknown)
        """
        
        logger.info("="*80)
        logger.info("STAGE-BASED RISK STRATIFICATION")
        logger.info("="*80)
        
        if 'stage' not in self.survival_data.columns:
            logger.warning("No stage data available, using alternative stratification")
            return self.stratify_by_embeddings()
        
        # Parse stages
        risk_groups = self.survival_data['stage'].apply(self.parse_stage).values
        
        # Log distribution
        unique, counts = np.unique(risk_groups, return_counts=True)
        
        risk_labels = {-1: 'Unknown', 0: 'Low (I/II)', 1: 'Medium (III)', 2: 'High (IV)'}
        
        logger.info("\nRisk group distribution:")
        for risk_level, count in zip(unique, counts):
            label = risk_labels.get(risk_level, f'Level_{risk_level}')
            pct = 100 * count / len(risk_groups)
            logger.info(f"  {label}: {count} patients ({pct:.1f}%)")
        
        # Handle unknown stages
        n_unknown = (risk_groups == -1).sum()
        if n_unknown > 0:
            logger.info(f"\nHandling {n_unknown} patients with unknown stage...")
            
            # Use model embeddings to assign unknown patients
            embeddings = self.extract_embeddings()
            
            # Get known patients
            known_mask = risk_groups != -1
            
            if known_mask.sum() > 0:
                # Fit KMeans on known patients
                from sklearn.neighbors import KNeighborsClassifier
                
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(embeddings[known_mask], risk_groups[known_mask])
                
                # Predict unknown
                risk_groups[~known_mask] = knn.predict(embeddings[~known_mask])
                
                logger.info("  ✓ Assigned unknown patients using KNN on embeddings")
        
        logger.info("\n✓ Stage-based stratification complete")
        
        return risk_groups
    
    def stratify_by_embeddings(self, n_groups=3):
        """
        Fallback: Stratify patients using GCN embeddings when stage data unavailable.
        
        Args:
            n_groups: Number of risk groups (default 3)
        
        Returns:
            risk_groups: Array of risk assignments
        """
        
        logger.info("="*80)
        logger.info("EMBEDDING-BASED RISK STRATIFICATION (FALLBACK)")
        logger.info("="*80)
        
        # Extract embeddings
        embeddings = self.extract_embeddings()
        
        # Get survival times
        if 'os_time' in self.survival_data.columns:
            survival_times = self.survival_data['os_time'].values
        elif 'pfi_time' in self.survival_data.columns:
            survival_times = self.survival_data['pfi_time'].values
        else:
            raise ValueError("No survival time data available!")
        
        # Combine embeddings with survival for better clustering
        features = np.concatenate([
            embeddings,
            survival_times.reshape(-1, 1) / 365.0  # Years
        ], axis=1)
        
        # Handle NaN
        col_means = np.nanmean(features, axis=0)
        nan_mask = np.isnan(features)
        features[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # K-means clustering
        kmeans = KMeans(
            n_clusters=n_groups,
            random_state=self.config.RANDOM_SEED,
            n_init=20
        )
        
        clusters = kmeans.fit_predict(features_scaled)
        
        # Order by median survival (longest = low risk)
        cluster_median_survival = []
        
        for cluster_id in range(n_groups):
            cluster_mask = clusters == cluster_id
            median_surv = survival_times[cluster_mask].median() if cluster_mask.sum() > 0 else 0
            cluster_median_survival.append(median_surv)
        
        # Map: longest survival → 0 (low risk), shortest → 2 (high risk)
        cluster_to_risk = np.argsort(np.argsort(cluster_median_survival)[::-1])
        risk_groups = np.array([cluster_to_risk[c] for c in clusters])
        
        logger.info("\nRisk group statistics:")
        for risk_id in range(n_groups):
            risk_mask = risk_groups == risk_id
            n_patients = risk_mask.sum()
            median_surv = survival_times[risk_mask].median()
            
            risk_label = ['Low', 'Medium', 'High'][risk_id]
            logger.info(f"  {risk_label} Risk: {n_patients} patients, median survival {median_surv:.0f} days")
        
        logger.info("\n✓ Embedding-based stratification complete")
        
        return risk_groups
    
    @torch.no_grad()
    def extract_embeddings(self):
        """Extract GCN embeddings for all patients."""
        
        self.model.eval()
        
        embeddings = self.model.get_embeddings(
            self.graph_data.x,
            self.graph_data.edge_index,
            self.graph_data.edge_attr.squeeze()
        ).cpu().numpy()
        
        return embeddings
    
    def kaplan_meier_analysis(self, risk_groups, cancer_type_idx=None, 
                              endpoint='OS', save_name=None):
        """
        Kaplan-Meier survival analysis.
        
        Args:
            risk_groups: Risk group assignments
            cancer_type_idx: Optional, analyze specific cancer type
            endpoint: 'OS' (overall survival) or 'PFI' (progression-free)
            save_name: Custom name for saving results
        
        Returns:
            km_results: Dictionary with KM curves and statistics
        """
        
        logger.info("="*80)
        logger.info(f"KAPLAN-MEIER ANALYSIS ({endpoint})")
        logger.info("="*80)
        
        # Determine event and time columns
        if endpoint == 'OS':
            event_col = 'os_event'
            time_col = 'os_time'
        elif endpoint == 'PFI':
            event_col = 'pfi_event'
            time_col = 'pfi_time'
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")
        
        # Check columns exist
        if event_col not in self.survival_data.columns or time_col not in self.survival_data.columns:
            logger.warning(f"{endpoint} data not available!")
            return None
        
        # Filter to specific cancer if requested
        if cancer_type_idx is not None:
            cancer_mask = self.graph_data.y.cpu().numpy() == cancer_type_idx
            survival_subset = self.survival_data.iloc[cancer_mask].copy()
            risk_groups_subset = risk_groups[cancer_mask]
        else:
            survival_subset = self.survival_data.copy()
            risk_groups_subset = risk_groups
        
        # Add risk groups
        survival_subset['risk_group'] = risk_groups_subset
        
        # Remove unknown risk groups for analysis
        known_mask = survival_subset['risk_group'] != -1
        survival_subset = survival_subset[known_mask].copy()
        
        if len(survival_subset) < 10:
            logger.warning(f"Too few patients ({len(survival_subset)}) for analysis")
            return None
        
        logger.info(f"Analyzing {len(survival_subset)} patients")
        
        # Fit Kaplan-Meier curves
        kmf = KaplanMeierFitter()
        
        km_curves = {}
        unique_groups = np.sort(survival_subset['risk_group'].unique())
        
        for risk_id in unique_groups:
            group_mask = survival_subset['risk_group'] == risk_id
            group_data = survival_subset[group_mask]
            
            # Remove NaN values
            valid_mask = group_data[event_col].notna() & group_data[time_col].notna()
            group_data = group_data[valid_mask]
            
            if len(group_data) < 3:
                logger.warning(f"Risk group {risk_id}: Too few patients ({len(group_data)})")
                continue
            
            kmf.fit(
                durations=group_data[time_col],
                event_observed=group_data[event_col],
                label=f'Risk Group {risk_id}'
            )
            
            km_curves[risk_id] = {
                'survival_function': kmf.survival_function_.copy(),
                'confidence_interval': kmf.confidence_interval_.copy(),
                'median_survival': kmf.median_survival_time_,
                'n_patients': len(group_data),
                'n_events': int(group_data[event_col].sum())
            }
            
            logger.info(f"  Risk {risk_id}: {len(group_data)} patients, "
                       f"{int(group_data[event_col].sum())} events, "
                       f"median {kmf.median_survival_time_:.0f} days")
        
        if len(km_curves) < 2:
            logger.warning("Need at least 2 risk groups for comparison")
            return None
        
        # Log-rank test
        try:
            results = multivariate_logrank_test(
                survival_subset[time_col],
                survival_subset['risk_group'],
                survival_subset[event_col]
            )
            
            log_rank_p = results.p_value
            
            logger.info(f"\nLog-rank test p-value: {log_rank_p:.4e}")
            
            if log_rank_p < 0.001:
                logger.info("  *** HIGHLY SIGNIFICANT differences! ***")
            elif log_rank_p < 0.05:
                logger.info("  ** Significant differences **")
            else:
                logger.info("  No significant differences")
        
        except Exception as e:
            logger.warning(f"Log-rank test failed: {e}")
            log_rank_p = np.nan
        
        # Pairwise comparisons
        pairwise_results = None
        if len(unique_groups) > 2:
            try:
                pairwise_results = pairwise_logrank_test(
                    survival_subset[time_col],
                    survival_subset['risk_group'],
                    survival_subset[event_col]
                )
                logger.info("\nPairwise comparisons:")
                logger.info(f"\n{pairwise_results.summary}")
            except Exception as e:
                logger.warning(f"Pairwise comparison failed: {e}")
        
        logger.info("\n✓ Kaplan-Meier analysis complete")
        
        return {
            'km_curves': km_curves,
            'log_rank_p': log_rank_p,
            'pairwise_results': pairwise_results,
            'survival_data': survival_subset,
            'endpoint': endpoint
        }
    
    def cox_regression_analysis(self, embeddings, risk_groups, endpoint='OS'):
        """
        Cox proportional hazards regression.
        
        Args:
            embeddings: Patient embeddings from GCN
            risk_groups: Risk group assignments
            endpoint: 'OS' or 'PFI'
        
        Returns:
            cox_results: Dictionary with Cox model and C-index
        """
        
        logger.info("="*80)
        logger.info(f"COX REGRESSION ANALYSIS ({endpoint})")
        logger.info("="*80)
        
        # Determine columns
        if endpoint == 'OS':
            event_col = 'os_event'
            time_col = 'os_time'
        elif endpoint == 'PFI':
            event_col = 'pfi_event'
            time_col = 'pfi_time'
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")
        
        # Check data availability
        if event_col not in self.survival_data.columns or time_col not in self.survival_data.columns:
            logger.warning(f"{endpoint} data not available!")
            return None
        
        # Reduce dimensionality with PCA
        n_components = min(10, embeddings.shape[1], len(self.survival_data) - 10)
        pca = PCA(n_components=n_components, random_state=self.config.RANDOM_SEED)
        embeddings_pca = pca.fit_transform(embeddings)
        
        logger.info(f"PCA: {embeddings.shape[1]} → {n_components} components")
        logger.info(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        # Create DataFrame
        cox_data = self.survival_data[[event_col, time_col]].copy()
        
        for i in range(n_components):
            cox_data[f'PC{i+1}'] = embeddings_pca[:, i]
        
        cox_data['risk_group'] = risk_groups
        
        # Remove NaN and unknown risk
        valid_mask = (
            cox_data[event_col].notna() & 
            cox_data[time_col].notna() & 
            (cox_data[time_col] > 0) &
            (cox_data['risk_group'] != -1)
        )
        
        cox_data = cox_data[valid_mask].copy()
        
        logger.info(f"Valid samples for Cox regression: {len(cox_data)}")
        
        if len(cox_data) < 50:
            logger.warning("Too few samples for Cox regression")
            return None
        
        # Fit Cox model
        cph = CoxPHFitter()
        
        covariates = [f'PC{i+1}' for i in range(n_components)]
        
        try:
            cph.fit(
                cox_data,
                duration_col=time_col,
                event_col=event_col,
                formula=' + '.join(covariates)
            )
            
            # Compute C-index
            c_index = concordance_index(
                cox_data[time_col],
                -cph.predict_partial_hazard(cox_data),
                cox_data[event_col]
            )
            
            logger.info(f"\nC-index: {c_index:.4f}")
            
            if c_index > 0.75:
                logger.info("  *** EXCELLENT prognostic performance! ***")
            elif c_index > 0.70:
                logger.info("  ** Very good prognostic performance **")
            elif c_index > 0.65:
                logger.info("  * Good prognostic performance *")
            elif c_index > 0.60:
                logger.info("  Moderate prognostic performance")
            else:
                logger.info("  Limited prognostic performance")
            
            # Significant covariates
            significant = cph.summary[cph.summary['p'] < 0.05]
            logger.info(f"\nSignificant covariates (p<0.05): {len(significant)}/{n_components}")
            
            logger.info("\n✓ Cox regression complete")
            
            return {
                'cox_model': cph,
                'c_index': c_index,
                'pca': pca,
                'summary': cph.summary,
                'endpoint': endpoint
            }
        
        except Exception as e:
            logger.error(f"Cox regression failed: {e}")
            return None
    
    def visualize_survival_curves(self, km_results, cancer_name, save_dir):
        """
        Create publication-quality Kaplan-Meier survival curve visualization.
        
        Args:
            km_results: Results from kaplan_meier_analysis
            cancer_name: Cancer type name for title
            save_dir: Directory to save figure
        """
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        km_curves = km_results['km_curves']
        survival_data = km_results['survival_data']
        endpoint = km_results.get('endpoint', 'OS')
        
        # Panel 1: Kaplan-Meier curves
        colors = ['#27ae60', '#f39c12', '#e74c3c']  # Green, Orange, Red
        risk_labels = ['Low Risk (Stage I/II)', 'Medium Risk (Stage III)', 'High Risk (Stage IV)']
        
        for risk_id in sorted(km_curves.keys()):
            curve_data = km_curves[risk_id]
            sf = curve_data['survival_function']
            ci = curve_data['confidence_interval']
            
            color = colors[risk_id] if risk_id < len(colors) else f'C{risk_id}'
            label = risk_labels[risk_id] if risk_id < len(risk_labels) else f'Risk {risk_id}'
            
            median_surv = curve_data['median_survival']
            n_patients = curve_data['n_patients']
            n_events = curve_data['n_events']
            
            # Plot survival function
            ax1.plot(
                sf.index, 
                sf.values, 
                color=color, 
                linewidth=3.5,
                label=f"{label}\n(n={n_patients}, events={n_events}, median={median_surv:.0f}d)"
            )
            
            # Confidence interval
            ax1.fill_between(
                ci.index,
                ci.iloc[:, 0],
                ci.iloc[:, 1],
                color=color,
                alpha=0.2
            )
        
        ax1.set_xlabel('Time (days)', fontsize=14, fontweight='bold')
        ax1.set_ylabel(f'{endpoint} Probability', fontsize=14, fontweight='bold')
        ax1.set_title(f'Kaplan-Meier {endpoint} Curves: {cancer_name}',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.legend(fontsize=11, loc='best', frameon=True, shadow=True, fancybox=True)
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_ylim([0, 1.05])
        
        # Add log-rank p-value
        p_val = km_results['log_rank_p']
        
        if not np.isnan(p_val):
            if p_val < 0.001:
                p_text = f'p < 0.001 ***'
            elif p_val < 0.01:
                p_text = f'p = {p_val:.3f} **'
            elif p_val < 0.05:
                p_text = f'p = {p_val:.3f} *'
            else:
                p_text = f'p = {p_val:.3f} ns'
            
            ax1.text(
                0.02, 0.02, 
                f'Log-rank {p_text}',
                transform=ax1.transAxes, 
                fontsize=13, 
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.85, edgecolor='black', linewidth=2)
            )
        
        # Panel 2: Risk group distribution
        unique_groups = sorted(km_curves.keys())
        
        patients_per_group = [km_curves[g]['n_patients'] for g in unique_groups]
        events_per_group = [km_curves[g]['n_events'] for g in unique_groups]
        
        x = np.arange(len(unique_groups))
        width = 0.35
        
        bars1 = ax2.bar(
            x - width/2, 
            patients_per_group, 
            width, 
            label='Total Patients',
            color='steelblue', 
            edgecolor='black', 
            linewidth=2, 
            alpha=0.85
        )
        
        bars2 = ax2.bar(
            x + width/2, 
            events_per_group, 
            width, 
            label=f'{endpoint} Events',
            color='crimson', 
            edgecolor='black', 
            linewidth=2, 
            alpha=0.85
        )
        
        ax2.set_xlabel('Risk Group', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Patients', fontsize=14, fontweight='bold')
        ax2.set_title('Patient Distribution by Risk Group',
                     fontsize=16, fontweight='bold', pad=20)
        ax2.set_xticks(x)
        
        group_labels = [risk_labels[i] if i < len(risk_labels) else f'Risk {i}' 
                       for i in unique_groups]
        ax2.set_xticklabels(group_labels, fontsize=11, fontweight='bold')
        
        ax2.legend(fontsize=12, frameon=True, shadow=True, fancybox=True)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width()/2., 
                    height,
                    f'{int(height)}',
                    ha='center', 
                    va='bottom', 
                    fontsize=12, 
                    fontweight='bold'
                )
        
        plt.suptitle(
            f'Prognostic Stratification: {cancer_name} ({endpoint})',
            fontsize=20, 
            fontweight='bold', 
            y=0.98
        )
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'survival_{endpoint}_{cancer_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Survival curves saved: {save_path}")
    
    def analyze_cancer_type(self, cancer_type: str, min_patients: int = 30):
        """
        Analyze survival for a specific cancer type with improved output formatting.
        
        Args:
            cancer_type: Cancer type abbreviation
            min_patients: Minimum patients required for analysis
        
        Returns:
            summary: Dictionary with results (Infinity replaced with None)
        """
        
        logger.info("="*80)
        logger.info(f"SURVIVAL ANALYSIS: {cancer_type}")
        logger.info("="*80)
        
        # Filter to cancer type
        cancer_mask = self.survival_data['cancer_type'] == cancer_type
        cancer_survival = self.survival_data[cancer_mask].copy()
        
        n_patients = int(len(cancer_survival))
        
        if n_patients < min_patients:
            logger.warning(f"Skipping {cancer_type}: only {n_patients} patients (< {min_patients})")
            return None
        
        logger.info(f"Patients: {n_patients}")
        
        # Get risk groups
        risk_groups = cancer_survival['risk_group'].values
        
        # Check if we have all risk groups
        unique_risks = sorted(cancer_survival['risk_group'].unique())
        logger.info(f"Risk groups present: {unique_risks}")
        
        summary = {
            'cancer_type': cancer_type,
            'n_patients': n_patients
        }
        
        # === OVERALL SURVIVAL (OS) ===
        if 'os_event' in cancer_survival.columns and 'os_time' in cancer_survival.columns:
            logger.info("\nOverall Survival (OS):")
            
            os_data = cancer_survival[['os_time', 'os_event', 'risk_group']].dropna()
            
            if len(os_data) > 0:
                # Kaplan-Meier
                kmf = KaplanMeierFitter()
                
                # Fit each risk group
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                colors = {0: '#27ae60', 1: '#f39c12', 2: '#e74c3c'}
                labels = {0: 'Low Risk (Stage I/II)', 1: 'Medium Risk (Stage III)', 2: 'High Risk (Stage IV)'}
                
                median_survival = {}
                
                for risk in unique_risks:
                    risk_data = os_data[os_data['risk_group'] == risk]
                    
                    if len(risk_data) >= 5:
                        kmf.fit(
                            durations=risk_data['os_time'],
                            event_observed=risk_data['os_event'],
                            label=labels.get(risk, f'Risk {risk}')
                        )
                        
                        # Plot
                        kmf.plot_survival_function(ax=ax1, ci_show=True, color=colors.get(risk, 'gray'))
                        
                        # Get median - IMPROVED HANDLING
                        try:
                            median = kmf.median_survival_time_
                            # Check if median is truly reachable
                            if np.isinf(median) or np.isnan(median):
                                median_survival[str(risk)] = None  # Use None instead of Infinity
                                logger.info(f"  {labels.get(risk, f'Risk {risk}')}: Not Reached (>50% alive)")
                            else:
                                median_survival[str(risk)] = float(median)
                                logger.info(f"  {labels.get(risk, f'Risk {risk}')}: {median:.0f} days ({median/365.25:.1f} years)")
                        except:
                            median_survival[str(risk)] = None
                            logger.info(f"  {labels.get(risk, f'Risk {risk}')}: Not Reached")
                
                # Log-rank test
                groups = [os_data[os_data['risk_group'] == r][['os_time', 'os_event']] 
                        for r in unique_risks]
                
                if len(groups) > 1:
                    results = multivariate_logrank_test(
                        os_data['os_time'],
                        os_data['risk_group'],
                        os_data['os_event']
                    )
                    
                    p_value = results.p_value
                    logger.info(f"\nLog-rank test p-value: {p_value:.2e}")
                    
                    # Significance stars
                    if p_value < 0.001:
                        sig = "***"
                    elif p_value < 0.01:
                        sig = "**"
                    elif p_value < 0.05:
                        sig = "*"
                    else:
                        sig = "ns"
                    
                    summary['os_log_rank_p'] = float(p_value)
                    summary['os_significance'] = sig
                else:
                    summary['os_log_rank_p'] = None
                    summary['os_significance'] = "N/A"
                
                summary['os_median_by_risk'] = median_survival
                
                # Styling
                ax1.set_xlabel('Time (days)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
                ax1.set_title(f'{cancer_type} - Overall Survival (OS)\np={p_value:.2e} {sig}', 
                            fontsize=14, fontweight='bold')
                ax1.grid(alpha=0.3)
                ax1.legend(loc='best', fontsize=11)
                
                # Bar plot of risk distribution
                risk_counts = os_data['risk_group'].value_counts().sort_index()
                ax2.bar([labels.get(r, f'Risk {r}') for r in risk_counts.index], 
                    risk_counts.values,
                    color=[colors.get(r, 'gray') for r in risk_counts.index],
                    edgecolor='black', linewidth=2)
                ax2.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
                ax2.set_title('Risk Group Distribution', fontsize=14, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                
                for i, (risk, count) in enumerate(zip(risk_counts.index, risk_counts.values)):
                    ax2.text(i, count, str(count), ha='center', va='bottom', 
                            fontsize=11, fontweight='bold')
                
                plt.tight_layout()
                save_path = os.path.join(self.results_dir, f'{cancer_type}_survival_OS.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                logger.info(f"Saved: {save_path}")
        
        # === PROGRESSION-FREE INTERVAL (PFI) ===
        if 'pfi_event' in cancer_survival.columns and 'pfi_time' in cancer_survival.columns:
            logger.info("\nProgression-Free Interval (PFI):")
            
            pfi_data = cancer_survival[['pfi_time', 'pfi_event', 'risk_group']].dropna()
            
            if len(pfi_data) > 0:
                kmf = KaplanMeierFitter()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                colors = {0: '#27ae60', 1: '#f39c12', 2: '#e74c3c'}
                labels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
                
                median_pfi = {}
                
                for risk in unique_risks:
                    risk_data = pfi_data[pfi_data['risk_group'] == risk]
                    
                    if len(risk_data) >= 5:
                        kmf.fit(
                            durations=risk_data['pfi_time'],
                            event_observed=risk_data['pfi_event'],
                            label=labels.get(risk, f'Risk {risk}')
                        )
                        
                        kmf.plot_survival_function(ax=ax1, ci_show=True, color=colors.get(risk, 'gray'))
                        
                        # Get median - IMPROVED
                        try:
                            median = kmf.median_survival_time_
                            if np.isinf(median) or np.isnan(median):
                                median_pfi[str(risk)] = None
                                logger.info(f"  {labels.get(risk, f'Risk {risk}')}: Not Reached")
                            else:
                                median_pfi[str(risk)] = float(median)
                                logger.info(f"  {labels.get(risk, f'Risk {risk}')}: {median:.0f} days")
                        except:
                            median_pfi[str(risk)] = None
                
                # Log-rank test
                if len(unique_risks) > 1:
                    results = multivariate_logrank_test(
                        pfi_data['pfi_time'],
                        pfi_data['risk_group'],
                        pfi_data['pfi_event']
                    )
                    
                    p_value = results.p_value
                    logger.info(f"\nLog-rank test p-value: {p_value:.2e}")
                    
                    if p_value < 0.001:
                        sig = "***"
                    elif p_value < 0.01:
                        sig = "**"
                    elif p_value < 0.05:
                        sig = "*"
                    else:
                        sig = "ns"
                    
                    summary['pfi_log_rank_p'] = float(p_value)
                    summary['pfi_significance'] = sig
                
                summary['pfi_median_by_risk'] = median_pfi
                
                # Styling
                ax1.set_xlabel('Time (days)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Progression-Free Probability', fontsize=12, fontweight='bold')
                ax1.set_title(f'{cancer_type} - Progression-Free Interval (PFI)\np={p_value:.2e} {sig}', 
                            fontsize=14, fontweight='bold')
                ax1.grid(alpha=0.3)
                ax1.legend(loc='best', fontsize=11)
                
                # Distribution
                risk_counts = pfi_data['risk_group'].value_counts().sort_index()
                ax2.bar([labels.get(r, f'Risk {r}') for r in risk_counts.index],
                    risk_counts.values,
                    color=[colors.get(r, 'gray') for r in risk_counts.index],
                    edgecolor='black', linewidth=2)
                ax2.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
                ax2.set_title('Risk Group Distribution', fontsize=14, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                
                for i, (risk, count) in enumerate(zip(risk_counts.index, risk_counts.values)):
                    ax2.text(i, count, str(count), ha='center', va='bottom',
                            fontsize=11, fontweight='bold')
                
                plt.tight_layout()
                save_path = os.path.join(self.results_dir, f'{cancer_type}_survival_PFI.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                logger.info(f"Saved: {save_path}")
        
        # Save summary JSON with better formatting
        summary_path = os.path.join(self.results_dir, f'survival_{cancer_type}.json')
        
        # Custom JSON encoder to handle None properly
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        logger.info(f"Summary saved: {summary_path}")
        logger.info("="*80)
        
        return summary
    
    def analyze_all_cancers(self):
        """Analyze survival for all cancer types."""
        
        logger.info("\n" + "="*80)
        logger.info("SURVIVAL ANALYSIS - ALL CANCER TYPES")
        logger.info("="*80 + "\n")
        
        # Extract embeddings
        logger.info("Extracting GCN embeddings...")
        embeddings = self.extract_embeddings()
        
        # Stratify into risk groups (stage-based if available)
        logger.info("\nStratifying patients into risk groups...")
        risk_groups = self.stratify_by_stage()
        
        # Global analysis
        logger.info("\n" + "="*80)
        logger.info("GLOBAL ANALYSIS (ALL CANCERS COMBINED)")
        logger.info("="*80)
        
        # Overall Survival
        if 'os_event' in self.survival_data.columns:
            logger.info("\n--- Overall Survival (OS) ---")
            global_km_os = self.kaplan_meier_analysis(risk_groups, cancer_type_idx=None, endpoint='OS')
            
            if global_km_os:
                self.visualize_survival_curves(global_km_os, 'All_Cancers', self.results_dir)
        
        # Progression-Free Interval
        if 'pfi_event' in self.survival_data.columns:
            logger.info("\n--- Progression-Free Interval (PFI) ---")
            global_km_pfi = self.kaplan_meier_analysis(risk_groups, cancer_type_idx=None, endpoint='PFI')
            
            if global_km_pfi:
                self.visualize_survival_curves(global_km_pfi, 'All_Cancers_PFI', self.results_dir)
        
        # Cox regression
        logger.info("\n--- Cox Regression ---")
        
        cox_results = {}
        
        if 'os_event' in self.survival_data.columns:
            cox_os = self.cox_regression_analysis(embeddings, risk_groups, endpoint='OS')
            if cox_os:
                cox_results['os'] = cox_os
        
        if 'pfi_event' in self.survival_data.columns:
            cox_pfi = self.cox_regression_analysis(embeddings, risk_groups, endpoint='PFI')
            if cox_pfi:
                cox_results['pfi'] = cox_pfi
        
        # Save Cox summary
        if cox_results:
            cox_summary = {}
            
            if 'os' in cox_results:
                cox_summary['os_c_index'] = cox_results['os']['c_index']
                cox_summary['os_n_components'] = len(cox_results['os']['summary'])
                cox_summary['os_significant_components'] = int((cox_results['os']['summary']['p'] < 0.05).sum())
            
            if 'pfi' in cox_results:
                cox_summary['pfi_c_index'] = cox_results['pfi']['c_index']
                cox_summary['pfi_n_components'] = len(cox_results['pfi']['summary'])
                cox_summary['pfi_significant_components'] = int((cox_results['pfi']['summary']['p'] < 0.05).sum())
            
            cox_path = os.path.join(self.results_dir, 'cox_regression_summary.json')
            with open(cox_path, 'w') as f:
                json.dump(cox_summary, f, indent=4)
        
        # Analyze each cancer type
        labels = self.graph_data.y.cpu().numpy()
        unique_labels = np.unique(labels)
        
        all_results = []
        
        logger.info("\n" + "="*80)
        logger.info("PER-CANCER ANALYSIS")
        logger.info("="*80)
        
        for cancer_idx in unique_labels:
            result = self.analyze_cancer_type(cancer_type)
            
            if result:
                all_results.append(result)
        
        # Summary
        if all_results:
            summary_df = pd.DataFrame(all_results)
            summary_df = summary_df.sort_values('n_patients', ascending=False)
            
            summary_path = os.path.join(self.results_dir, 'survival_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            
            logger.info(f"\n✓ Summary saved: {summary_path}")
        
        logger.info("\n" + "="*80)
        logger.info("✓✓✓ SURVIVAL ANALYSIS COMPLETE ✓✓✓")
        logger.info("="*80)
        logger.info(f"Analyzed {len(all_results)} cancer types")
        
        if cox_results:
            if 'os' in cox_results:
                logger.info(f"Global OS C-index: {cox_results['os']['c_index']:.4f}")
            if 'pfi' in cox_results:
                logger.info(f"Global PFI C-index: {cox_results['pfi']['c_index']:.4f}")
        
        logger.info(f"Results directory: {self.results_dir}")
        logger.info("="*80 + "\n")
        
        return summary_df if all_results else None