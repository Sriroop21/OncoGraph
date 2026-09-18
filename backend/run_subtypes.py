import sys
sys.path.append('.')

import torch
import os
from backend.configs.config import Config
from backend.utils.graph_builder import GraphBuilder
from backend.models.model_factory import ModelFactory
from backend.analysis.subtypes import BiologyInformedSubtypeDiscovery

def main():
    print("\n" + "="*60)
    print("SUBTYPE DISCOVERY")
    print("="*60 + "\n")
    
    config = Config()
    device = config.DEVICE
    
    print("Loading data")
    graph_builder = GraphBuilder(config)
    graph_data, _ = graph_builder.load_graph(config.GRAPH_PATH)
    
    print("Loading model")
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
    
    analyzer = BiologyInformedSubtypeDiscovery(model, graph_data, config, device)
    summary = analyzer.analyze_all_cancer_types()
    
    print("\nCOMPLETE")
    print(f"Results: {analyzer.results_dir}\n")

if __name__ == "__main__":
    main()