import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
import logging
from typing import Tuple, Dict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Advanced data preprocessing pipeline for pan-cancer RNA-Seq analysis.
    
    Handles:
    - Gene expression data loading and normalization
    - Clinical/survival data integration
    - Missing value imputation
    - Stratified train/val/test splitting
    - Proper survival data formatting for downstream analysis
    
    Output: Comprehensive processed dataset with genes, labels, and clinical outcomes
    """
    
    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.gene_names = None
        self.cancer_types = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load gene expression and clinical data from disk.
        
        Returns:
            gene_expr: DataFrame with genes as rows, patients as columns
            clinical: DataFrame with patient clinical/survival information
        """
        logger.info("="*80)
        logger.info("LOADING DATA")
        logger.info("="*80)
        
        # Load gene expression
        logger.info(f"Loading gene expression from: {self.config.GENE_EXPR_PATH}")
        gene_expr = pd.read_csv(
            self.config.GENE_EXPR_PATH, 
            sep='\t', 
            index_col=0,
            low_memory=False
        )
        logger.info(f"  Gene expression shape: {gene_expr.shape}")
        logger.info(f"  Genes: {gene_expr.shape[0]}, Patients: {gene_expr.shape[1]}")
        
        # Load clinical data
        logger.info(f"Loading clinical data from: {self.config.CLINICAL_PATH}")
        clinical = pd.read_csv(
            self.config.CLINICAL_PATH,
            sep='\t',
            low_memory=False
        )
        logger.info(f"  Clinical data shape: {clinical.shape}")
        logger.info(f"  Patients with clinical data: {len(clinical)}")
        
        # Verify required columns exist
        required_cols = ['sample', 'cancer type abbreviation']
        missing_cols = [col for col in required_cols if col not in clinical.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in clinical data: {missing_cols}")
        
        logger.info(f"✓ Data loaded successfully")
        
        return gene_expr, clinical
    
    def merge_datasets(self, gene_expr: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
        """
        Merge gene expression with clinical/survival data.
        
        Args:
            gene_expr: Gene expression matrix (genes × patients)
            clinical: Clinical metadata (patients × features)
        
        Returns:
            merged: Combined dataset (patients × [genes + clinical])
        """
        logger.info("="*80)
        logger.info("MERGING DATASETS")
        logger.info("="*80)
        
        # Transpose gene expression (genes as columns, patients as rows)
        logger.info("Transposing gene expression matrix...")
        gene_expr_T = gene_expr.T
        gene_expr_T.index.name = 'sample'
        gene_expr_T = gene_expr_T.reset_index()
        logger.info(f"  Transposed shape: {gene_expr_T.shape}")
        
        # Select clinical columns
        # Core columns
        clinical_cols = ['sample', 'cancer type abbreviation']
        
        # Survival columns (check which exist)
        survival_cols = ['OS', 'OS.time', 'DSS', 'DSS.time', 'PFI', 'PFI.time', 
                        'vital_status', 'ajcc_pathologic_tumor_stage']
        
        available_survival_cols = [col for col in survival_cols if col in clinical.columns]
        
        logger.info(f"Available survival columns: {available_survival_cols}")
        
        clinical_subset = clinical[clinical_cols + available_survival_cols].copy()
        
        logger.info(f"Clinical subset shape: {clinical_subset.shape}")
        
        # Inner join on patient ID
        logger.info("Performing inner join on patient IDs...")
        merged = pd.merge(
            gene_expr_T, 
            clinical_subset, 
            on='sample', 
            how='inner'
        )
        
        logger.info(f"  Merged dataset shape: {merged.shape}")
        logger.info(f"  Patients retained: {len(merged)}")
        logger.info(f"  Features: {merged.shape[1] - len(clinical_cols) - len(available_survival_cols)} genes")
        
        # Check cancer type distribution
        cancer_dist = merged['cancer type abbreviation'].value_counts()
        logger.info(f"  Unique cancer types: {len(cancer_dist)}")
        logger.info(f"  Most common: {cancer_dist.index[0]} ({cancer_dist.values[0]} patients)")
        logger.info(f"  Least common: {cancer_dist.index[-1]} ({cancer_dist.values[-1]} patients)")
        
        logger.info("✓ Datasets merged successfully")
        
        return merged
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in gene expression and clinical features.
        
        Strategy:
        - Gene expression: Median imputation (robust to outliers)
        - Clinical: Keep as-is (handled later in specific analyses)
        
        Args:
            data: Merged dataset
        
        Returns:
            data: Dataset with imputed gene expression values
        """
        logger.info("="*80)
        logger.info("HANDLING MISSING VALUES")
        logger.info("="*80)
        
        # Remove duplicate columns
        n_cols_before = len(data.columns)
        data = data.loc[:, ~data.columns.duplicated(keep='first')]
        n_cols_after = len(data.columns)
        
        if n_cols_before != n_cols_after:
            logger.warning(f"  Removed {n_cols_before - n_cols_after} duplicate columns")
        
        # Identify column types
        metadata_cols = ['sample', 'cancer type abbreviation', 'OS', 'OS.time', 
                        'DSS', 'DSS.time', 'PFI', 'PFI.time', 
                        'vital_status', 'ajcc_pathologic_tumor_stage']
        
        # Gene columns = everything else
        gene_cols = [col for col in data.columns if col not in metadata_cols]
        
        logger.info(f"Gene columns: {len(gene_cols)}")
        logger.info(f"Metadata columns: {len([c for c in metadata_cols if c in data.columns])}")
        
        # Extract gene features
        features = data[gene_cols].values
        
        # Check missing values
        n_missing = np.isnan(features).sum()
        pct_missing = 100 * n_missing / features.size
        
        logger.info(f"Missing values in gene data: {n_missing:,} ({pct_missing:.2f}%)")
        
        if n_missing > 0:
            logger.info("Imputing missing values (median strategy)...")
            features_imputed = self.imputer.fit_transform(features)
            logger.info("✓ Imputation complete")
        else:
            features_imputed = features
            logger.info("✓ No missing values detected")
        
        # Reconstruct dataframe
        result = pd.DataFrame(features_imputed, columns=gene_cols)
        
        # Add back metadata
        result.insert(0, 'sample', data['sample'].values)
        
        for col in metadata_cols[1:]:  # Skip 'sample'
            if col in data.columns:
                result[col] = data[col].values
        
        logger.info(f"Final shape after imputation: {result.shape}")
        
        return result
    
    def normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score normalization of gene expression features.
        
        Formula: z = (x - μ) / σ
        
        Args:
            data: Dataset with imputed values
        
        Returns:
            data: Dataset with normalized gene expression
        """
        logger.info("="*80)
        logger.info("NORMALIZING FEATURES (Z-SCORE)")
        logger.info("="*80)
        
        # Identify metadata columns
        metadata_cols = ['sample', 'cancer type abbreviation', 'OS', 'OS.time', 
                        'DSS', 'DSS.time', 'PFI', 'PFI.time',
                        'vital_status', 'ajcc_pathologic_tumor_stage']
        
        # Gene columns
        gene_cols = [col for col in data.columns if col not in metadata_cols]
        
        logger.info(f"Normalizing {len(gene_cols)} gene features...")
        
        # Extract features
        features = data[gene_cols].values
        
        # Z-score normalization
        features_normalized = self.scaler.fit_transform(features)
        
        logger.info(f"  Original - Mean: {features.mean():.4f}, Std: {features.std():.4f}")
        logger.info(f"  Normalized - Mean: {features_normalized.mean():.4f}, Std: {features_normalized.std():.4f}")
        
        # Reconstruct
        result = pd.DataFrame(features_normalized, columns=gene_cols)
        
        # Add metadata
        # Replace lines 248-250 with:
        for i, col in enumerate(['sample'] + [c for c in metadata_cols[1:] if c in data.columns]):
            result.insert(i, col, data[col].values)
        
        logger.info("✓ Normalization complete")
        
        return result
    
    def encode_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode cancer type labels to integer indices.
        
        Args:
            data: Dataset with cancer type strings
        
        Returns:
            data: Dataset with encoded 'label' column
        """
        logger.info("="*80)
        logger.info("ENCODING LABELS")
        logger.info("="*80)
        
        data = data.copy()
        
        # Encode cancer types
        data['label'] = self.label_encoder.fit_transform(data['cancer type abbreviation'])
        
        self.cancer_types = self.label_encoder.classes_
        
        logger.info(f"Encoded {len(self.cancer_types)} cancer types:")
        
        # Show encoding
        for idx, cancer in enumerate(self.cancer_types[:10]):  # First 10
            count = (data['label'] == idx).sum()
            logger.info(f"  {idx:2d}: {cancer:6s} ({count:4d} patients)")
        
        if len(self.cancer_types) > 10:
            logger.info(f"  ... and {len(self.cancer_types) - 10} more")
        
        logger.info("✓ Labels encoded")
        
        return data
    
    def stratified_split(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Stratified train/val/test split preserving class distribution.
        
        Ensures each cancer type is proportionally represented in all splits.
        
        Args:
            data: Full dataset
        
        Returns:
            Dictionary with train_idx, val_idx, test_idx
        """
        logger.info("="*80)
        logger.info("STRATIFIED SPLIT")
        logger.info("="*80)
        
        indices = np.arange(len(data))
        labels = data['label'].values
        
        logger.info(f"Total samples: {len(indices)}")
        logger.info(f"Train ratio: {self.config.TRAIN_SPLIT:.1%}")
        logger.info(f"Val ratio: {self.config.VAL_SPLIT:.1%}")
        logger.info(f"Test ratio: {self.config.TEST_SPLIT:.1%}")
        
        # First split: train vs (val+test)
        sss1 = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=(self.config.VAL_SPLIT + self.config.TEST_SPLIT),
            random_state=self.config.RANDOM_SEED
        )
        train_idx, temp_idx = next(sss1.split(indices, labels))
        
        # Second split: val vs test
        temp_labels = labels[temp_idx]
        val_ratio = self.config.VAL_SPLIT / (self.config.VAL_SPLIT + self.config.TEST_SPLIT)
        
        sss2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=(1 - val_ratio),
            random_state=self.config.RANDOM_SEED
        )
        val_idx_temp, test_idx_temp = next(sss2.split(temp_idx, temp_labels))
        val_idx = temp_idx[val_idx_temp]
        test_idx = temp_idx[test_idx_temp]
        
        logger.info(f"✓ Split complete:")
        logger.info(f"  Train: {len(train_idx):5d} ({100*len(train_idx)/len(indices):.1f}%)")
        logger.info(f"  Val:   {len(val_idx):5d} ({100*len(val_idx)/len(indices):.1f}%)")
        logger.info(f"  Test:  {len(test_idx):5d} ({100*len(test_idx)/len(indices):.1f}%)")
        
        # Verify stratification
        for split_name, split_idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
            unique, counts = np.unique(labels[split_idx], return_counts=True)
            logger.info(f"  {split_name}: {len(unique)} classes represented")
        
        return {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
    
    def prepare_survival_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare survival data for downstream analysis.
        
        Converts OS, OS.time, PFI, etc. into standard format.
        
        Args:
            data: Full dataset
        
        Returns:
            survival_df: Clean survival DataFrame
        """
        logger.info("="*80)
        logger.info("PREPARING SURVIVAL DATA")
        logger.info("="*80)
        
        survival_cols = {
            'sample': 'patient_id',
            'cancer type abbreviation': 'cancer_type',
            'OS': 'os_event',
            'OS.time': 'os_time',
            'DSS': 'dss_event',
            'DSS.time': 'dss_time',
            'PFI': 'pfi_event',
            'PFI.time': 'pfi_time',
            'vital_status': 'vital_status',
            'ajcc_pathologic_tumor_stage': 'stage'
        }
        
        # Select available columns
        available_cols = {k: v for k, v in survival_cols.items() if k in data.columns}
        
        survival_df = data[list(available_cols.keys())].copy()
        survival_df = survival_df.rename(columns=available_cols)
        
        logger.info(f"Survival data columns: {list(survival_df.columns)}")
        
        # Handle OS (Overall Survival)
        if 'os_event' in survival_df.columns and 'os_time' in survival_df.columns:
            # Convert to proper format
            survival_df['os_event'] = survival_df['os_event'].astype(float)
            survival_df['os_time'] = survival_df['os_time'].astype(float)
            
            n_events = survival_df['os_event'].sum()
            n_total = len(survival_df)
            
            logger.info(f"  Overall Survival (OS):")
            logger.info(f"    Events (deaths): {int(n_events)} ({100*n_events/n_total:.1f}%)")
            logger.info(f"    Censored (alive): {int(n_total - n_events)} ({100*(n_total-n_events)/n_total:.1f}%)")
            logger.info(f"    Median time: {survival_df['os_time'].median():.0f} days")
        
        # Handle PFI (Progression-Free Interval)
        if 'pfi_event' in survival_df.columns and 'pfi_time' in survival_df.columns:
            survival_df['pfi_event'] = survival_df['pfi_event'].astype(float)
            survival_df['pfi_time'] = survival_df['pfi_time'].astype(float)
            
            n_events = survival_df['pfi_event'].sum()
            n_total = len(survival_df)
            
            logger.info(f"  Progression-Free Interval (PFI):")
            logger.info(f"    Events: {int(n_events)} ({100*n_events/n_total:.1f}%)")
            logger.info(f"    Median time: {survival_df['pfi_time'].median():.0f} days")
        
        # Handle staging
        if 'stage' in survival_df.columns:
            stage_dist = survival_df['stage'].value_counts()
            logger.info(f"  Stage distribution:")
            for stage, count in stage_dist.head(5).items():
                logger.info(f"    {stage}: {count}")
        
        logger.info("✓ Survival data prepared")
        
        return survival_df
    
    def extract_features_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract gene features and cancer type labels.
        
        Args:
            data: Full dataset
        
        Returns:
            features: Gene expression matrix (n_patients, n_genes)
            labels: Cancer type labels (n_patients,)
        """
        logger.info("="*80)
        logger.info("EXTRACTING FEATURES AND LABELS")
        logger.info("="*80)
        
        # Identify gene columns
        metadata_cols = ['sample', 'cancer type abbreviation', 'label', 
                        'OS', 'OS.time', 'DSS', 'DSS.time', 'PFI', 'PFI.time',
                        'vital_status', 'ajcc_pathologic_tumor_stage']
        
        gene_cols = [col for col in data.columns if col not in metadata_cols]
        
        self.gene_names = gene_cols
        
        # Extract
        features = data[gene_cols].values.astype(np.float32)
        labels = data['label'].values.astype(np.int64)
        
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Gene count: {len(self.gene_names)}")
        
        logger.info("✓ Extraction complete")
        
        return features, labels
    
    def process_pipeline(self) -> Dict:
        """
        Complete preprocessing pipeline.
        
        Executes all preprocessing steps in order:
        1. Load data
        2. Merge datasets
        3. Handle missing values
        4. Normalize features
        5. Encode labels
        6. Stratified split
        7. Extract features/labels
        8. Prepare survival data
        
        Returns:
            Dictionary with all processed data and metadata
        """
        logger.info("\n" + "="*80)
        logger.info("DATA PREPROCESSING PIPELINE")
        logger.info("="*80 + "\n")
        
        # Load
        gene_expr, clinical = self.load_data()
        
        # Merge
        merged = self.merge_datasets(gene_expr, clinical)
        
        # Handle missing values
        merged = self.handle_missing_values(merged)
        
        # Normalize
        merged = self.normalize_features(merged)
        
        # Encode labels
        merged = self.encode_labels(merged)
        
        # Split
        split_indices = self.stratified_split(merged)
        
        # Extract features and labels
        features, labels = self.extract_features_labels(merged)
        
        # Prepare survival data
        survival_data = self.prepare_survival_data(merged)
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"✓ Total patients: {len(features)}")
        logger.info(f"✓ Total genes: {len(self.gene_names)}")
        logger.info(f"✓ Cancer types: {len(self.cancer_types)}")
        logger.info(f"✓ Survival data: {len(survival_data)} patients")
        logger.info("="*80 + "\n")
        
        return {
            'features': features,
            'labels': labels,
            'survival_data': survival_data,  # ✅ NOW A PROPER DATAFRAME!
            'split_indices': split_indices,
            'sample_ids': merged['sample'].values,
            'cancer_types': self.cancer_types,
            'gene_names': self.gene_names,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'imputer': self.imputer
        }
    
    def save_processed_data(self, processed_data: Dict, save_path: str):
        """
        Save processed data to disk.
        
        Args:
            processed_data: Dictionary from process_pipeline()
            save_path: Path to save .pt file
        """
        logger.info(f"Saving processed data to: {save_path}")
        
        # Create directory if doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save(processed_data, save_path)
        
        # Get file size
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        
        logger.info(f"✓ Data saved successfully ({file_size_mb:.1f} MB)")
    
    def load_processed_data(self, load_path: str) -> Dict:
        """
        Load processed data from disk.
        
        Args:
            load_path: Path to .pt file
        
        Returns:
            Dictionary with all processed data
        """
        logger.info(f"Loading processed data from: {load_path}")
        
        data = torch.load(load_path, weights_only=False)
        
        # Restore instance variables
        self.cancer_types = data['cancer_types']
        self.gene_names = data['gene_names']
        self.label_encoder = data['label_encoder']
        self.scaler = data['scaler']
        self.imputer = data['imputer']
        
        logger.info("✓ Data loaded successfully")
        
        return data


# ============================================================================
# MAIN SCRIPT - Run this file directly to process data
# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, project_root)
    
    from backend.configs.config import Config
    
    print("\n" + "="*80)
    print("ONCOGRAPH DATA PREPROCESSING")
    print("="*80 + "\n")
    
    # Initialize
    config = Config()
    processor = DataProcessor(config)
    
    # Run pipeline
    processed_data = processor.process_pipeline()
    
    # Save
    processor.save_processed_data(processed_data, config.PROCESSED_DATA_PATH)
    
    print("\n" + "="*80)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"Saved to: {config.PROCESSED_DATA_PATH}")
    print("="*80 + "\n")