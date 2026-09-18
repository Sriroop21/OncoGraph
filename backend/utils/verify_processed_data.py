import torch
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.configs.config import Config

def verify_processed_data():
    """Comprehensive verification of processed data."""
    
    print("\n" + "="*80)
    print("VERIFYING PROCESSED DATA")
    print("="*80 + "\n")
    
    config = Config()
    
    # Load processed data
    print(f"Loading from: {config.PROCESSED_DATA_PATH}")
    data = torch.load(config.PROCESSED_DATA_PATH, weights_only=False)
    
    print("\n" + "="*80)
    print("1. CHECKING DATA STRUCTURE")
    print("="*80)
    
    # Check keys
    expected_keys = ['features', 'labels', 'survival_data', 'split_indices', 
                     'sample_ids', 'cancer_types', 'gene_names', 
                     'label_encoder', 'scaler', 'imputer']
    
    print("\nExpected keys:")
    for key in expected_keys:
        if key in data:
            print(f"  ✓ {key}")
        else:
            print(f"  ✗ {key} - MISSING!")
    
    print("\n" + "="*80)
    print("2. CHECKING FEATURES")
    print("="*80)
    
    features = data['features']
    print(f"\nShape: {features.shape}")
    print(f"Type: {features.dtype}")
    print(f"Patients: {features.shape[0]}")
    print(f"Genes: {features.shape[1]}")
    print(f"Mean: {features.mean():.4f}")
    print(f"Std: {features.std():.4f}")
    print(f"Min: {features.min():.4f}")
    print(f"Max: {features.max():.4f}")
    print(f"NaN values: {np.isnan(features).sum()}")
    
    if np.isnan(features).sum() > 0:
        print("  ⚠️ WARNING: Found NaN values!")
    else:
        print("  ✓ No NaN values")
    
    print("\n" + "="*80)
    print("3. CHECKING LABELS")
    print("="*80)
    
    labels = data['labels']
    cancer_types = data['cancer_types']
    
    print(f"\nShape: {labels.shape}")
    print(f"Type: {labels.dtype}")
    print(f"Unique classes: {len(np.unique(labels))}")
    print(f"Min label: {labels.min()}")
    print(f"Max label: {labels.max()}")
    
    print("\nCancer type distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for idx, (label_idx, count) in enumerate(zip(unique[:10], counts[:10])):
        cancer_name = cancer_types[label_idx]
        print(f"  {label_idx:2d}. {cancer_name:6s}: {count:4d} patients ({100*count/len(labels):.1f}%)")
    
    if len(unique) > 10:
        print(f"  ... and {len(unique) - 10} more cancer types")
    
    print("\n" + "="*80)
    print("4. CHECKING SURVIVAL DATA")
    print("="*80)
    
    survival_data = data['survival_data']
    
    print(f"\nType: {type(survival_data)}")
    
    if isinstance(survival_data, pd.DataFrame):
        print(f"Shape: {survival_data.shape}")
        print(f"\nColumns: {list(survival_data.columns)}")
        
        print("\nColumn details:")
        for col in survival_data.columns:
            print(f"  • {col}:")
            print(f"      Type: {survival_data[col].dtype}")
            print(f"      Non-null: {survival_data[col].notna().sum()}/{len(survival_data)}")
            
            if col in ['os_event', 'dss_event', 'pfi_event']:
                n_events = survival_data[col].sum()
                print(f"      Events: {int(n_events)} ({100*n_events/len(survival_data):.1f}%)")
            
            elif col in ['os_time', 'dss_time', 'pfi_time']:
                print(f"      Median: {survival_data[col].median():.0f} days")
                print(f"      Range: {survival_data[col].min():.0f} - {survival_data[col].max():.0f} days")
            
            elif col == 'stage':
                stage_counts = survival_data[col].value_counts()
                print(f"      Stages: {list(stage_counts.head(3).index)}")
        
        # ✅ KEY CHECK: Verify OS and PFI data exists
        print("\n" + "-"*80)
        print("SURVIVAL DATA VALIDATION:")
        
        has_os = 'os_event' in survival_data.columns and 'os_time' in survival_data.columns
        has_pfi = 'pfi_event' in survival_data.columns and 'pfi_time' in survival_data.columns
        has_stage = 'stage' in survival_data.columns
        
        if has_os:
            print("  ✓ Overall Survival (OS) data present")
        else:
            print("  ✗ Overall Survival (OS) data MISSING!")
        
        if has_pfi:
            print("  ✓ Progression-Free Interval (PFI) data present")
        else:
            print("  ✗ Progression-Free Interval (PFI) data MISSING!")
        
        if has_stage:
            print("  ✓ Stage data present")
        else:
            print("  ⚠️ Stage data missing (optional)")
        
    else:
        print(f"  ⚠️ WARNING: survival_data is {type(survival_data)}, expected DataFrame!")
    
    print("\n" + "="*80)
    print("5. CHECKING SPLIT INDICES")
    print("="*80)
    
    split_indices = data['split_indices']
    
    train_idx = split_indices['train_idx']
    val_idx = split_indices['val_idx']
    test_idx = split_indices['test_idx']
    
    total = len(train_idx) + len(val_idx) + len(test_idx)
    
    print(f"\nTrain: {len(train_idx)} ({100*len(train_idx)/total:.1f}%)")
    print(f"Val:   {len(val_idx)} ({100*len(val_idx)/total:.1f}%)")
    print(f"Test:  {len(test_idx)} ({100*len(test_idx)/total:.1f}%)")
    print(f"Total: {total}")
    
    # Check for overlap
    overlap_train_val = len(set(train_idx) & set(val_idx))
    overlap_train_test = len(set(train_idx) & set(test_idx))
    overlap_val_test = len(set(val_idx) & set(test_idx))
    
    if overlap_train_val > 0 or overlap_train_test > 0 or overlap_val_test > 0:
        print("\n  ⚠️ WARNING: Found overlapping indices between splits!")
    else:
        print("\n  ✓ No overlap between splits")
    
    print("\n" + "="*80)
    print("6. CHECKING GENE NAMES")
    print("="*80)
    
    gene_names = data['gene_names']
    
    print(f"\nTotal genes: {len(gene_names)}")
    print(f"First 10 genes: {gene_names[:10]}")
    print(f"Last 10 genes: {gene_names[-10:]}")
    
    # Check for duplicates
    duplicates = len(gene_names) - len(set(gene_names))
    if duplicates > 0:
        print(f"\n  ⚠️ WARNING: Found {duplicates} duplicate gene names!")
    else:
        print("\n  ✓ No duplicate gene names")
    
    print("\n" + "="*80)
    print("7. CHECKING SAMPLE IDS")
    print("="*80)
    
    sample_ids = data['sample_ids']
    
    print(f"\nTotal samples: {len(sample_ids)}")
    print(f"Example IDs: {sample_ids[:5]}")
    
    # Check match with features
    if len(sample_ids) == len(features):
        print(f"\n  ✓ Sample IDs match feature count")
    else:
        print(f"\n  ✗ Mismatch: {len(sample_ids)} IDs vs {len(features)} features!")
    
    # Check match with survival data
    if isinstance(survival_data, pd.DataFrame):
        if len(survival_data) == len(sample_ids):
            print(f"  ✓ Survival data matches sample count")
        else:
            print(f"  ✗ Mismatch: {len(survival_data)} survival vs {len(sample_ids)} samples!")
    
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    # Overall checks
    checks = []
    
    # Check 1: All required keys present
    all_keys_present = all(key in data for key in expected_keys)
    checks.append(("All keys present", all_keys_present))
    
    # Check 2: No NaN in features
    no_nan_features = np.isnan(features).sum() == 0
    checks.append(("No NaN in features", no_nan_features))
    
    # Check 3: Labels valid
    labels_valid = labels.min() >= 0 and labels.max() < len(cancer_types)
    checks.append(("Labels valid", labels_valid))
    
    # Check 4: Survival data is DataFrame
    survival_is_df = isinstance(survival_data, pd.DataFrame)
    checks.append(("Survival data is DataFrame", survival_is_df))
    
    # Check 5: Has OS data
    has_os_data = 'os_event' in survival_data.columns if survival_is_df else False
    checks.append(("Has OS survival data", has_os_data))
    
    # Check 6: No split overlap
    no_overlap = (overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0)
    checks.append(("No split overlap", no_overlap))
    
    # Check 7: Counts match
    counts_match = (len(sample_ids) == len(features) == len(labels))
    checks.append(("Sample counts match", counts_match))
    
    print("\n")
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
    
    all_passed = all(passed for _, passed in checks)
    
    print("\n" + "="*80)
    if all_passed:
        print("✓✓✓ ALL CHECKS PASSED! DATA IS READY! ✓✓✓")
    else:
        print("⚠️⚠️⚠️ SOME CHECKS FAILED! REVIEW ABOVE! ⚠️⚠️⚠️")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    verify_processed_data()