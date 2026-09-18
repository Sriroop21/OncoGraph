import sys
sys.path.append('.')

import torch
import os
from backend.configs.config import Config
from backend.utils.graph_builder import GraphBuilder
from backend.models.model_factory import ModelFactory
from backend.analysis.biomarkers import CorrectedBiomarkerDiscovery

def main():
    print("\n" + "="*80)
    print("BIOMARKER DISCOVERY")
    print("FDR Correction")
    print("="*80 + "\n")
    
    config = Config()
    device = config.DEVICE
    
    # Load
    print("Loading data...")
    graph_builder = GraphBuilder(config)
    graph_data, _ = graph_builder.load_graph(config.GRAPH_PATH)
    
    print("Loading model...")
    model = ModelFactory.create_model(
        'gat',
        graph_data.num_node_features,
        len(torch.unique(graph_data.y)),
        config,
        device
    )
    
    checkpoint = torch.load(
        os.path.join(config.RESULTS_PATH, 'checkpoints', 'best_model.pt'),
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Ready\n")
    
    print("CORRECTIONS:")
    print("FDR multiple testing correction (Benjamini-Hochberg)")
    print("Balanced ensemble weights (stats reduced 40% → 25%)")
    print("Known cancer gene database (ESR1, PGR, ERBB2, etc.)")
    print("Known gene boost (+15% for cancer-specific, +10% general)")
    print("Robust percentile normalization")
    print("\nExpected: ESR1, PGR, ERBB2 in top 10 for BRCA")
    print("Expected: 40-50% known gene validation rate")
    print("\nEstimated time: 2-3 hours for all 33 cancers\n")
    
    confirm = input("Continue? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("Cancelled.")
        return
    
    # Run
    analyzer = CorrectedBiomarkerDiscovery(model, graph_data, config, device)
    
    summary = analyzer.analyze_all_cancers(
        n_samples=100,
        top_k=50
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results: {analyzer.results_dir}\n")

if __name__ == "__main__":
    main()