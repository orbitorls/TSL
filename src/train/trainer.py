"""Core training logic for TSL-51 models.

This module contains the training loop and related functions extracted from
train_tsl51_v3.py for better modularity.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Optional, Dict, Any
import logging

from .config import TrainingConfig
from src.core.models import MODEL_REGISTRY as MODEL_CLASSES

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for TSL-51 models.
    
    This class handles the training loop, validation, and model checkpointing.
    """
    
    def __init__(self, config: TrainingConfig, device: str):
        """Initialize trainer.
        
        Args:
            config: Training configuration
            device: Device to use ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        
    def setup_model(self, input_dim: int, num_classes: int):
        """Setup model, optimizer, scheduler, and scaler.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
        """
        # Get model class
        model_class = MODEL_CLASSES.get(self.config.model, MODEL_CLASSES['gru'])
        
        # Create model
        self.model = model_class(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4,
        )
        
        # Setup scheduler - will update steps_per_epoch when train_loader is available
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.epochs,
            steps_per_epoch=1,  # Placeholder, updated in train()
        )
        
        # Setup gradient scaler for mixed precision
        if self.config.use_amp:
            try:
                from torch.amp.grad_scaler import GradScaler
                self.scaler = GradScaler()
            except ImportError:
                try:
                    from torch.cuda.amp import GradScaler
                    self.scaler = GradScaler()
                except ImportError:
                    self.scaler = None
                    logger.warning("AMP not available, training without mixed precision")
    
    def train_epoch(self, train_loader, criterion):
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.use_amp and self.scaler is not None:
                # Mixed precision training
                try:
                    with torch.amp.autocast(device_type=self.device):
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                    
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.config.gradient_clip_value is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_value
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except Exception:
                    # Fallback to regular training if AMP fails
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    
                    if self.config.gradient_clip_value is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_value
                        )
                    
                    self.optimizer.step()
            else:
                # Regular training
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                if self.config.gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_value
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (loss, accuracy, predictions, labels)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)
    
    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        classes,
        fold_idx: int = 0
    ) -> Dict[str, Any]:
        """Train the model on given train/val split.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            classes: Class names
            fold_idx: Current fold index
            
        Returns:
            Dictionary containing training results
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Windows compatibility
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Update scheduler with correct steps_per_epoch based on actual data
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.epochs,
            steps_per_epoch=len(train_loader),
        )
        
        # Setup loss function with class weighting
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Reset best metrics
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        best_state = None  # Initialize to track best model state
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        # Training loop
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion)
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader, criterion)
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Fold {fold_idx} Epoch {epoch+1}/{self.config.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping check
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                
                # Save best model state
                best_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                }
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state['model_state_dict'])
        
        # Final evaluation
        from sklearn.metrics import f1_score, precision_score, recall_score
        val_loss, val_acc, val_preds, val_labels = self.validate(val_loader, criterion)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_precision = precision_score(val_labels, val_preds, average='weighted')
        val_recall = recall_score(val_labels, val_preds, average='weighted')
        
        results = {
            'fold': fold_idx,
            'history': history,
            'best_val_acc': self.best_val_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1_score': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'model_state': best_state,
        }
        
        return results
