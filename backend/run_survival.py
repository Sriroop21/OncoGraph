import sys
import os

# --- ROBUST PATH SETUP ---
def setup_project_root():
    """
    Automatically finds the project root by looking for the 'backend' directory.
    This works regardless of where this script is located in your project.
    """
    # Start from the directory containing this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = current_dir

    # Walk up directory levels (max 5 levels) to find 'backend'
    for _ in range(5):
        if os.path.exists(os.path.join(root_path, 'backend')):
            # Found the root! Add it to sys.path
            if root_path not in sys.path:
                sys.path.insert(0, root_path)
            return
        
        # Move up one level
        parent = os.path.dirname(root_path)
        if parent == root_path: # Hit the filesystem root
            break
        root_path = parent

    print("Warning: Could not automatically locate the 'backend' folder.")
    print("Please ensure this script is inside the project directory.")

setup_project_root()
# -------------------------

import torch
from backend.configs.config import Config
from backend.utils.graph_builder import GraphBuilder
from backend.models.model_factory import ModelFactory
from backend.analysis.survival import SurvivalAnalysisPro

def main():
    print("\n" + "="*80)
    print("OBJECTIVE 4: PROFESSIONAL SURVIVAL ANALYSIS")
    print("Using REAL TCGA Clinical Data (Stages, OS, PFI)")
    print("="*80 + "\n")
    
    config = Config()
    device = config.DEVICE
    
    # Load data
    print("Loading graph data...")
    graph_builder = GraphBuilder(config)
    graph_data, _ = graph_builder.load_graph(config.GRAPH_PATH)
    
    # Load model
    print("Loading trained model...")
    # NOTE: Ensure num_node_features and classes match your saved model
    # You might need to check if graph_data.y exists and has shape
    num_classes = len(torch.unique(graph_data.y)) if hasattr(graph_data, 'y') and graph_data.y is not None else 2
    
    model = ModelFactory.create_model(
        'gat',
        graph_data.num_node_features,
        num_classes,
        config,
        device
    )
    
    checkpoint_path = os.path.join(config.RESULTS_PATH, 'checkpoints', 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded\n")
    
    print("="*80)
    print("ANALYSIS PIPELINE:")
    print("="*80)
    print("  1. Load REAL survival data (OS, PFI, stages)")
    print("  2. Stage-based risk stratification (I/II → Low, III → Medium, IV → High)")
    print("  3. Kaplan-Meier curves for OS and PFI")
    print("  4. Log-rank statistical testing")
    print("  5. Cox proportional hazards regression")
    print("  6. C-index evaluation")
    print("  7. Per-cancer analysis (33 types)")
    print("\nEstimated time: 20-30 minutes")
    print("="*80 + "\n")
    
    confirm = input("Continue? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("Cancelled.")
        return
    
    # Run analysis
    analyzer = SurvivalAnalysisPro(model, graph_data, config, device)
    summary = analyzer.analyze_all_cancers()
    
    print("\n" + "="*80)
    print("SURVIVAL ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results: {analyzer.results_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()