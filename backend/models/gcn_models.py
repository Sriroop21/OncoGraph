import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.nn import BatchNorm, LayerNorm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual connection block for GCN layers."""
    
    def __init__(self, in_channels, out_channels, conv_layer, use_batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.conv = conv_layer
        self.batch_norm = BatchNorm(out_channels) if use_batch_norm else nn.Identity()
        
        # Residual connection (if dimensions don't match, use linear projection)
        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x, edge_index, edge_weight=None):
        identity = self.residual(x)
        
        if edge_weight is not None:
            out = self.conv(x, edge_index, edge_weight=edge_weight)
        else:
            out = self.conv(x, edge_index)
        
        out = self.batch_norm(out)
        out = out + identity  # Residual connection
        
        return out


class GCNEncoder(nn.Module):
    """
    Advanced Graph Convolutional Network encoder with residual connections.
    Supports multiple GCN architectures: GCN, GAT, GraphSAGE.
    """
    
    def __init__(self, in_channels, hidden_channels, num_layers=3,dropout=0.5, conv_type='gcn', use_batch_norm=True,attention_heads=4):
        super(GCNEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GCN layers with residual connections
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_channels
            out_dim = hidden_channels
            
            if conv_type == 'gcn':
                conv = GCNConv(in_dim, out_dim)
            elif conv_type == 'gat':
                # Multi-head attention
                conv = GATConv(in_dim, out_dim // attention_heads, 
                             heads=attention_heads, dropout=dropout, concat=True)
            elif conv_type == 'sage':
                conv = SAGEConv(in_dim, out_dim)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            
            self.convs.append(conv)
            
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_channels))
            else:
                self.batch_norms.append(nn.Identity())
        
        # Residual connections for each layer
        self.residuals = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels) if i > 0 else nn.Identity()
            for i in range(num_layers)
        ])
        
        logger.info(f"GCN Encoder initialized: {conv_type.upper()}, "
                   f"{num_layers} layers, hidden_dim={hidden_channels}")
    
    def forward(self, x, edge_index, edge_weight=None):
        # Initial projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GCN layers with residual connections
        for i, (conv, batch_norm, residual) in enumerate(zip(self.convs, self.batch_norms, self.residuals)):
            identity = residual(x)
            
            # Graph convolution
            if edge_weight is not None and self.conv_type == 'gcn':
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            
            # Batch normalization
            x = batch_norm(x)
            
            # Residual connection
            x = x + identity
            
            # Activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class OncoGraphClassifier(nn.Module):
    """
    Complete OncoGraph model for pan-cancer classification.
    Includes encoder, classifier head, and explainability hooks.
    """
    
    def __init__(self, num_features, num_classes, hidden_dim=512, 
                 num_layers=3, dropout=0.5, conv_type='gcn',
                 use_batch_norm=True, attention_heads=4):
        super(OncoGraphClassifier, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Graph encoder
        self.encoder = GCNEncoder(
            in_channels=num_features,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            conv_type=conv_type,
            use_batch_norm=use_batch_norm,
            attention_heads=attention_heads
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Store intermediate activations for explainability
        self.node_embeddings = None
        self.attention_weights = None
        
        logger.info(f"OncoGraphClassifier initialized: "
                   f"{num_features} features -> {num_classes} classes")
    
    def forward(self, x, edge_index, edge_weight=None, return_embeddings=False):
        # Encode graph
        embeddings = self.encoder(x, edge_index, edge_weight)
        
        # Store for explainability
        self.node_embeddings = embeddings.detach()
        
        # Classify
        logits = self.classifier(embeddings)
        
        if return_embeddings:
            return logits, embeddings
        
        return logits
    
    def get_embeddings(self, x, edge_index, edge_weight=None):
        """Extract node embeddings for visualization."""
        self.eval()
        with torch.no_grad():
            embeddings = self.encoder(x, edge_index, edge_weight)
        return embeddings
    
    def predict_with_confidence(self, x, edge_index, edge_weight=None):
        """Get predictions with confidence scores."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index, edge_weight)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
        
        return predictions, confidences, probs


class EnsembleGCN(nn.Module):
    """
    Ensemble of multiple GCN architectures for robust predictions.
    Combines GCN, GAT, and GraphSAGE.
    """
    
    def __init__(self, num_features, num_classes, hidden_dim=512,
                 num_layers=3, dropout=0.5):
        super(EnsembleGCN, self).__init__()
        
        # Multiple model architectures
        self.gcn_model = OncoGraphClassifier(
            num_features, num_classes, hidden_dim, num_layers, 
            dropout, conv_type='gcn'
        )
        
        self.gat_model = OncoGraphClassifier(
            num_features, num_classes, hidden_dim, num_layers,
            dropout, conv_type='gat', attention_heads=4
        )
        
        self.sage_model = OncoGraphClassifier(
            num_features, num_classes, hidden_dim, num_layers,
            dropout, conv_type='sage'
        )
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
        logger.info("Ensemble GCN initialized with 3 architectures")
    
    def forward(self, x, edge_index, edge_weight=None):
        # Get predictions from each model
        logits_gcn = self.gcn_model(x, edge_index, edge_weight)
        logits_gat = self.gat_model(x, edge_index, edge_weight)
        logits_sage = self.sage_model(x, edge_index, edge_weight)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        logits = (weights[0] * logits_gcn + 
                 weights[1] * logits_gat + 
                 weights[2] * logits_sage)
        
        return logits


class GCNWithUncertainty(nn.Module):
    """
    GCN with Monte Carlo Dropout for uncertainty estimation.
    Useful for anomaly detection (Objective 5).
    """
    
    def __init__(self, num_features, num_classes, hidden_dim=512,
                 num_layers=3, dropout=0.5, conv_type='gcn'):
        super(GCNWithUncertainty, self).__init__()
        
        self.base_model = OncoGraphClassifier(
            num_features, num_classes, hidden_dim, num_layers,
            dropout, conv_type
        )
        
        self.num_mc_samples = 10  # Monte Carlo samples
        
        logger.info("GCN with Uncertainty Estimation initialized")
    
    def forward(self, x, edge_index, edge_weight=None):
        return self.base_model(x, edge_index, edge_weight)
    
    def predict_with_uncertainty(self, x, edge_index, edge_weight=None):
        """
        Predict with uncertainty using Monte Carlo Dropout.
        Returns mean prediction and uncertainty (variance).
        """
        self.train()  # Keep dropout active
        
        predictions = []
        for _ in range(self.num_mc_samples):
            with torch.no_grad():
                logits = self.base_model(x, edge_index, edge_weight)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Stack predictions
        predictions = torch.stack(predictions)  # (num_samples, num_nodes, num_classes)
        
        # Mean prediction
        mean_probs = predictions.mean(dim=0)
        
        # Uncertainty (variance across samples)
        uncertainty = predictions.var(dim=0).mean(dim=1)  # Per-node uncertainty
        
        # Final prediction
        confidences, preds = torch.max(mean_probs, dim=1)
        
        self.eval()
        
        return preds, confidences, uncertainty


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model):
    """Xavier initialization for better convergence."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (GCNConv, GATConv, SAGEConv)):
            if hasattr(m, 'lin'):
                nn.init.xavier_uniform_(m.lin.weight)