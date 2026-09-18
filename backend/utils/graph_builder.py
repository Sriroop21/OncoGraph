import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import logging
from typing import Dict, Tuple
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Advanced graph construction for patient constellation network.
    Implements k-NN with multiple similarity metrics and adaptive edge weighting.
    """
    
    def __init__(self, config):
        self.config = config
        self.edge_index = None
        self.edge_weights = None
        self.node_embeddings = None
        
    def build_knn_graph(self, features: np.ndarray, k: int = 5, 
                       metric: str = 'cosine') -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct k-NN graph using efficient ball-tree algorithm.
        
        Args:
            features: Node feature matrix (n_samples, n_features)
            k: Number of nearest neighbors
            metric: Distance metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            edge_index: Edge connectivity (2, num_edges)
            edge_weights: Edge weights based on similarity
        """
        logger.info(f"Building k-NN graph with k={k}, metric={metric}")
        
        n_samples = features.shape[0]
        
        # Handle cosine similarity separately (requires normalization)
        if metric == 'cosine':
            # Compute pairwise cosine similarity
            logger.info("Computing cosine similarity matrix...")
            similarity_matrix = cosine_similarity(features)
            
            # For each node, find k nearest neighbors (excluding self)
            edge_list = []
            edge_weight_list = []
            
            for i in tqdm(range(n_samples), desc="Building edges"):
                # Get similarity scores for node i
                similarities = similarity_matrix[i]
                
                # Exclude self-loop and get top k
                similarities[i] = -np.inf
                top_k_indices = np.argpartition(similarities, -k)[-k:]
                top_k_similarities = similarities[top_k_indices]
                
                # Add edges (bidirectional)
                for j, sim in zip(top_k_indices, top_k_similarities):
                    edge_list.append([i, j])
                    edge_weight_list.append(sim)
                    
        else:
            # Use sklearn's NearestNeighbors for other metrics
            logger.info(f"Fitting NearestNeighbors with {metric} metric...")
            nbrs = NearestNeighbors(
                n_neighbors=k+1,  # +1 because it includes the point itself
                metric=metric,
                algorithm='auto',
                n_jobs=-1
            ).fit(features)
            
            logger.info("Finding k-nearest neighbors...")
            distances, indices = nbrs.kneighbors(features)
            
            # Remove self-loops (first column)
            indices = indices[:, 1:]
            distances = distances[:, 1:]
            
            # Convert distances to similarities (inverse distance)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            similarities = 1.0 / (distances + epsilon)
            
            # Build edge list
            edge_list = []
            edge_weight_list = []
            
            for i in range(n_samples):
                for j, sim in zip(indices[i], similarities[i]):
                    edge_list.append([i, j])
                    edge_weight_list.append(sim)
        
        # Convert to numpy arrays
        edge_index = np.array(edge_list, dtype=np.int64).T
        edge_weights = np.array(edge_weight_list, dtype=np.float32)
        
        # Normalize edge weights to [0, 1]
        edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-8)
        
        logger.info(f"Graph constructed: {n_samples} nodes, {len(edge_weights)} edges")
        logger.info(f"Edge weight stats - Min: {edge_weights.min():.4f}, "
                   f"Max: {edge_weights.max():.4f}, Mean: {edge_weights.mean():.4f}")
        
        return edge_index, edge_weights
    
    def make_graph_undirected(self, edge_index: np.ndarray, 
                            edge_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert directed graph to undirected by adding reverse edges.
        Average weights for duplicate edges.
        """
        logger.info("Converting to undirected graph...")
        
        # Create reverse edges
        reverse_edges = edge_index[[1, 0], :]
        
        # Combine original and reverse edges
        all_edges = np.concatenate([edge_index, reverse_edges], axis=1)
        all_weights = np.concatenate([edge_weights, edge_weights])
        
        # Remove duplicate edges and average weights
        edges_dict = {}
        for i in range(all_edges.shape[1]):
            edge = tuple(sorted([all_edges[0, i], all_edges[1, i]]))
            weight = all_weights[i]
            
            if edge in edges_dict:
                edges_dict[edge].append(weight)
            else:
                edges_dict[edge] = [weight]
        
        # Reconstruct edge_index and edge_weights
        unique_edges = []
        avg_weights = []
        
        for edge, weights in edges_dict.items():
            unique_edges.append([edge[0], edge[1]])
            unique_edges.append([edge[1], edge[0]])  # Both directions
            avg_weight = np.mean(weights)
            avg_weights.extend([avg_weight, avg_weight])
        
        edge_index = np.array(unique_edges, dtype=np.int64).T
        edge_weights = np.array(avg_weights, dtype=np.float32)
        
        logger.info(f"Undirected graph: {edge_index.shape[1]} edges")
        
        return edge_index, edge_weights
    
    def add_self_loops(self, edge_index: np.ndarray, edge_weights: np.ndarray, 
                      num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Add self-loops with weight 1.0 to all nodes."""
        logger.info("Adding self-loops...")
        
        self_loops = np.array([np.arange(num_nodes), np.arange(num_nodes)], dtype=np.int64)
        self_weights = np.ones(num_nodes, dtype=np.float32)
        
        edge_index = np.concatenate([edge_index, self_loops], axis=1)
        edge_weights = np.concatenate([edge_weights, self_weights])
        
        logger.info(f"Total edges with self-loops: {edge_index.shape[1]}")
        
        return edge_index, edge_weights
    
    def compute_graph_statistics(self, edge_index: np.ndarray, num_nodes: int) -> Dict:
        """Compute graph topology statistics."""
        logger.info("Computing graph statistics...")
        
        # Degree distribution
        degrees = np.bincount(edge_index[0], minlength=num_nodes)
        
        stats = {
            'num_nodes': num_nodes,
            'num_edges': edge_index.shape[1],
            'avg_degree': degrees.mean(),
            'max_degree': degrees.max(),
            'min_degree': degrees.min(),
            'degree_std': degrees.std(),
            'density': edge_index.shape[1] / (num_nodes * (num_nodes - 1))
        }
        
        logger.info("Graph Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        return stats
    
    def create_pyg_data(self, features: np.ndarray, labels: np.ndarray,
                       edge_index: np.ndarray, edge_weights: np.ndarray,
                       split_indices: Dict, survival_data: np.ndarray = None) -> Data:
        """
        Create PyTorch Geometric Data object.
        
        Args:
            features: Node features (n_nodes, n_features)
            labels: Node labels (n_nodes,)
            edge_index: Edge connectivity (2, n_edges)
            edge_weights: Edge weights (n_edges,)
            split_indices: Dict with train/val/test indices
            survival_data: Survival information (n_nodes, 4) [OS, OS.time, PFI, PFI.time]
            
        Returns:
            PyTorch Geometric Data object
        """
        logger.info("Creating PyTorch Geometric Data object...")
        
        # Convert to tensors
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        
        # Create masks
        num_nodes = features.shape[0]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[split_indices['train_idx']] = True
        val_mask[split_indices['val_idx']] = True
        test_mask[split_indices['test_idx']] = True
        
        # Create Data object
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        # Add survival data if provided
        # Replace lines 241-242 with:
        if survival_data is not None:
            if hasattr(survival_data, 'values'):
                numeric_cols = ['os_event', 'os_time', 'pfi_event', 'pfi_time']
                available = [c for c in numeric_cols if c in survival_data.columns]
                data.survival = torch.tensor(
                    survival_data[available].values.astype(np.float32),
                    dtype=torch.float
                )
            else:
                data.survival = torch.tensor(
                    np.array(survival_data, dtype=np.float32),
                    dtype=torch.float
                )
        
        logger.info(f"PyG Data created: {data}")
        
        return data
    
    def build_graph_pipeline(self, processed_data: Dict) -> Data:
        """
        Complete graph construction pipeline.
        
        Args:
            processed_data: Dictionary from DataProcessor
            
        Returns:
            PyTorch Geometric Data object
        """
        logger.info("="*60)
        logger.info("Starting Graph Construction Pipeline")
        logger.info("="*60)
        
        features = processed_data['features']
        labels = processed_data['labels']
        split_indices = processed_data['split_indices']
        survival_data = processed_data.get('survival_data', None)
        
        # Step 1: Build k-NN graph
        edge_index, edge_weights = self.build_knn_graph(
            features,
            k=self.config.K_NEIGHBORS,
            metric=self.config.SIMILARITY_METRIC
        )
        
        # Step 2: Make undirected
        edge_index, edge_weights = self.make_graph_undirected(edge_index, edge_weights)
        
        # Step 3: Add self-loops
        edge_index, edge_weights = self.add_self_loops(
            edge_index, edge_weights, num_nodes=features.shape[0]
        )
        
        # Step 4: Compute statistics
        stats = self.compute_graph_statistics(edge_index, num_nodes=features.shape[0])
        
        # Step 5: Create PyG Data
        graph_data = self.create_pyg_data(
            features, labels, edge_index, edge_weights, 
            split_indices, survival_data
        )
        
        logger.info("="*60)
        logger.info("Graph Construction Complete")
        logger.info("="*60)
        
        return graph_data, stats
    
    def save_graph(self, graph_data: Data, stats: Dict, save_path: str):
        """Save graph data and statistics."""
        logger.info(f"Saving graph to {save_path}")
        
        save_dict = {
            'graph_data': graph_data,
            'stats': stats
        }
        
        torch.save(save_dict, save_path)
        logger.info("Graph saved successfully")
    
    def load_graph(self, load_path: str) -> Tuple[Data, Dict]:
        """Load graph data from disk."""
        logger.info(f"Loading graph from {load_path}")
        
        save_dict = torch.load(load_path, weights_only=False)
        
        logger.info("Graph loaded successfully")
        return save_dict['graph_data'], save_dict['stats']