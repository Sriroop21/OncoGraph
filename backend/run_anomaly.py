import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from backend.configs.config import Config
from backend.utils.graph_builder import GraphBuilder
from backend.models.model_factory import ModelFactory
from backend.analysis.anomaly_detection import AnomalyDetectionProV3

def main():
    print("\n" + "="*80)
    print("OBJECTIVE 5: ANOMALY DETECTION V3 (CALIBRATED)")
    print("="*80)
    print("\nKEY IMPROVEMENTS:")
    print("Meta-learned ensemble (optimized on validation)")
    print("Class statistics from train+val (better generalization)")
    print("Calibrated thresholds (prevents overfitting)")
    print("Error-focused optimization")
    print("\nTARGET: 75-85% error detection on test")
    print("="*80 + "\n")
    
    config = Config()
    device = config.DEVICE
    
    print("Loading model and data...")
    graph_builder = GraphBuilder(config)
    graph_data, _ = graph_builder.load_graph(config.GRAPH_PATH)
    
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
    print("✓ Ready\n")
    
    confirm = input("Run calibrated analysis? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Cancelled.")
        return
    
    detector = AnomalyDetectionProV3(model, graph_data, config, device)
    summary = detector.run_complete_analysis()
    
    print("\n" + "="*80)
    print("✓✓✓ CALIBRATED ANALYSIS COMPLETE ✓✓✓")
    print("="*80)
    print("\nFinal Results:")
    print(summary.to_string(index=False))
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()