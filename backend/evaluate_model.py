import sys
sys.path.append('.')

import torch
from backend.configs.config import Config
from backend.utils.graph_builder import GraphBuilder
from backend.models.model_factory import ModelFactory
from backend.evaluation.evaluator import ModelEvaluator

def main():
    print("\n" + "="*60)
    print("ONCOGRAPH MODEL EVALUATION")
    print("="*60 + "\n")
    
    # Configuration
    config = Config()
    device = config.DEVICE
    
    # Load graph data
    print("Loading graph data...")
    graph_builder = GraphBuilder(config)
    graph_data, stats = graph_builder.load_graph(config.GRAPH_PATH)
    
    num_features = graph_data.num_node_features
    num_classes = len(torch.unique(graph_data.y))
    
    print(f"✓ Graph loaded\n")
    
    # Create model
    print("Creating model...")
    model = ModelFactory.create_model(
        model_type='gat',
        num_features=num_features,
        num_classes=num_classes,
        config=config,
        device=device
    )
    
    # Load best checkpoint
    checkpoint_path = os.path.join(config.RESULTS_PATH, 'checkpoints', 'best_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}\n")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, graph_data, config, device)
    
    # Run complete evaluation
    results = evaluator.run_complete_evaluation(mask_name='test')
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {config.RESULTS_PATH}")
    print("="*60 + "\n")

if __name__ == "__main__":
    import os
    main()