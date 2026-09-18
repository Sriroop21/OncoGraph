import sys
sys.path.append('.')

import torch
from backend.configs.config import Config
from backend.utils.graph_builder import GraphBuilder
from backend.models.model_factory import ModelFactory
from backend.training.trainer import OncoGraphTrainer

def main():
    print("\n" + "="*60)
    print("ONCOGRAPH MODEL TRAINING")
    print("="*60 + "\n")
    
    # Configuration
    config = Config()
    device = config.DEVICE
    
    print(f"Device: {device}")
    print(f"Model: GAT (Graph Attention Network)")
    print(f"Hidden dim: {config.HIDDEN_DIM}")
    print(f"Layers: {config.NUM_GCN_LAYERS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Max epochs: {config.NUM_EPOCHS}")
    print(f"Patience: {config.PATIENCE}\n")
    
    # Load graph data
    print("Loading graph data...")
    graph_builder = GraphBuilder(config)
    graph_data, stats = graph_builder.load_graph(config.GRAPH_PATH)
    
    num_features = graph_data.num_node_features
    num_classes = len(torch.unique(graph_data.y))
    
    print(f"Graph loaded: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    print(f"Features: {num_features}, Classes: {num_classes}\n")
    
    # Create model
    print("Creating model...")
    model = ModelFactory.create_model(
        model_type='gat',  # Using GAT for best performance
        num_features=num_features,
        num_classes=num_classes,
        config=config,
        device=device
    )
    print()
    
    # Create trainer
    trainer = OncoGraphTrainer(
        model=model,
        graph_data=graph_data,
        config=config,
        device=device
    )
    
    # Train
    results = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Best Validation Accuracy: {results['best_val_acc']:.4f}")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Final Test Accuracy: {results['test_acc']:.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()