import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, Tuple
import time
from collections import defaultdict
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=20, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(f"Early stopping triggered at epoch {epoch}")
            logger.info(f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
            return True
        
        return False


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics = defaultdict(list)
        self.epoch_start_time = None
    
    def update(self, metric_name, value):
        self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name):
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return np.mean(self.metrics[metric_name])
        return 0.0
    
    def get_all_averages(self):
        return {name: self.get_average(name) for name in self.metrics.keys()}


class OncoGraphTrainer:
    """
    Advanced trainer for OncoGraph models.
    Handles training loop, validation, early stopping, and checkpointing.
    """
    
    def __init__(self, model, graph_data, config, device='cpu'):
        self.model = model
        self.graph_data = graph_data.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        if hasattr(config, 'OPTIMIZER') and config.OPTIMIZER == 'adamw':
            self.optimizer = AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            self.optimizer = Adam(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.PATIENCE,
            mode='max'
        )
        
        # Metrics tracker
        self.train_tracker = MetricsTracker()
        self.val_tracker = MetricsTracker()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        logger.info("Trainer initialized")
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"Device: {device}")
    
    def compute_loss(self, logits, labels, mask):
        """Compute cross-entropy loss for masked nodes."""
        return F.cross_entropy(logits[mask], labels[mask])
    
    def compute_accuracy(self, logits, labels, mask):
        """Compute accuracy for masked nodes."""
        preds = logits[mask].argmax(dim=1)
        correct = (preds == labels[mask]).sum().item()
        total = mask.sum().item()
        return correct / total if total > 0 else 0.0
    
    def train_epoch(self):
        """Single training epoch."""
        self.model.train()
        self.train_tracker.reset()
        
        # Forward pass
        logits = self.model(
            self.graph_data.x,
            self.graph_data.edge_index,
            self.graph_data.edge_attr.squeeze()
        )
        
        # Compute loss and accuracy
        train_mask = self.graph_data.train_mask
        loss = self.compute_loss(logits, self.graph_data.y, train_mask)
        acc = self.compute_accuracy(logits, self.graph_data.y, train_mask)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Track metrics
        self.train_tracker.update('loss', loss.item())
        self.train_tracker.update('acc', acc)
        
        return loss.item(), acc
    
    @torch.no_grad()
    def validate(self):
        """Validation step."""
        self.model.eval()
        self.val_tracker.reset()
        
        # Forward pass
        logits = self.model(
            self.graph_data.x,
            self.graph_data.edge_index,
            self.graph_data.edge_attr.squeeze()
        )
        
        # Compute loss and accuracy
        val_mask = self.graph_data.val_mask
        loss = self.compute_loss(logits, self.graph_data.y, val_mask)
        acc = self.compute_accuracy(logits, self.graph_data.y, val_mask)
        
        # Track metrics
        self.val_tracker.update('loss', loss.item())
        self.val_tracker.update('acc', acc)
        
        return loss.item(), acc
    
    @torch.no_grad()
    def test(self):
        """Test evaluation."""
        self.model.eval()
        
        logits = self.model(
            self.graph_data.x,
            self.graph_data.edge_index,
            self.graph_data.edge_attr.squeeze()
        )
        
        test_mask = self.graph_data.test_mask
        acc = self.compute_accuracy(logits, self.graph_data.y, test_mask)
        
        # Get predictions for detailed analysis
        preds = logits[test_mask].argmax(dim=1)
        labels = self.graph_data.y[test_mask]
        
        return acc, preds, labels
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.RESULTS_PATH, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']
    
    def train(self, num_epochs=None):
        """
        Complete training loop.
        
        Args:
            num_epochs: Number of epochs (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.config.NUM_EPOCHS
        
        logger.info("="*60)
        logger.info("Starting Training")
        logger.info("="*60)
        logger.info(f"Total epochs: {num_epochs}")
        logger.info(f"Train samples: {self.graph_data.train_mask.sum().item()}")
        logger.info(f"Val samples: {self.graph_data.val_mask.sum().item()}")
        logger.info(f"Test samples: {self.graph_data.test_mask.sum().item()}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Logging
            epoch_time = time.time() - epoch_start
            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:03d} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s"
                )
            
            # Early stopping
            if self.early_stopping(val_acc, epoch):
                break
        
        total_time = time.time() - start_time
        
        # Final test evaluation
        test_acc, test_preds, test_labels = self.test()
        
        logger.info("="*60)
        logger.info("Training Complete")
        logger.info("="*60)
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        logger.info(f"Final test accuracy: {test_acc:.4f}")
        
        # Save training history
        self.save_training_history()
        
        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'test_acc': test_acc,
            'history': self.history
        }
    
    def save_training_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.config.RESULTS_PATH, 'training_history.json')
        os.makedirs(self.config.RESULTS_PATH, exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        logger.info(f"Training history saved to {history_path}")