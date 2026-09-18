import torch
from backend.models.gcn_models import (
    OncoGraphClassifier, 
    EnsembleGCN, 
    GCNWithUncertainty,
    count_parameters,
    initialize_weights
)
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating and managing models."""
    
    @staticmethod
    def create_model(model_type, num_features, num_classes, config, device='cpu'):
        """
        Create model based on type.
        
        Args:
            model_type: 'gcn', 'gat', 'sage', 'ensemble', 'uncertainty'
            num_features: Number of input features
            num_classes: Number of output classes
            config: Configuration object
            device: Device to place model on
        
        Returns:
            Initialized model
        """
        logger.info(f"Creating model: {model_type}")
        
        if model_type in ['gcn', 'gat', 'sage']:
            model = OncoGraphClassifier(
                num_features=num_features,
                num_classes=num_classes,
                hidden_dim=config.HIDDEN_DIM,
                num_layers=config.NUM_GCN_LAYERS,
                dropout=config.DROPOUT,
                conv_type=model_type
            )
        
        elif model_type == 'ensemble':
            model = EnsembleGCN(
                num_features=num_features,
                num_classes=num_classes,
                hidden_dim=config.HIDDEN_DIM,
                num_layers=config.NUM_GCN_LAYERS,
                dropout=config.DROPOUT
            )
        
        elif model_type == 'uncertainty':
            model = GCNWithUncertainty(
                num_features=num_features,
                num_classes=num_classes,
                hidden_dim=config.HIDDEN_DIM,
                num_layers=config.NUM_GCN_LAYERS,
                dropout=config.DROPOUT,
                conv_type='gat'  # Use GAT for uncertainty model
            )
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Initialize weights
        initialize_weights(model)
        
        # Move to device
        model = model.to(device)
        
        # Log parameters
        num_params = count_parameters(model)
        logger.info(f"Model parameters: {num_params:,}")
        
        return model
    
    @staticmethod
    def save_model(model, path, optimizer=None, epoch=None, metrics=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_type': model.__class__.__name__
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load_model(path, model, optimizer=None, device='cpu'):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', None)
        metrics = checkpoint.get('metrics', None)
        
        logger.info(f"Model loaded from {path}")
        
        return model, optimizer, epoch, metrics