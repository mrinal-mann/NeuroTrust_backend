"""
NeuroTrust Model Trainer - FIXED VERSION
Handles training the GRU + RELANFIS model on uploaded datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Tuple, Any
import logging
from tqdm import tqdm

from models.neurotrust_model import NeuroTrustModel, create_neurotrust_model

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trainer class for NeuroTrust model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def create_data_loaders(self, 
                          X: np.ndarray, 
                          y: np.ndarray, 
                          batch_size: int = 32,
                          test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders"""
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        return train_loader, val_loader
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Only calculate AUC if we have both classes
        if len(np.unique(y_true)) > 1:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        else:
            metrics['auc'] = 0.0
        
        return metrics
    
    def train_epoch(self, 
                   model: NeuroTrustModel, 
                   train_loader: DataLoader, 
                   optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Dict[str, float]:
        """Train for one epoch"""
        
        model.train()
        total_loss = 0.0
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output, debug_info = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Collect metrics
            total_loss += loss.item()
            
            # Convert to numpy for metrics calculation
            predictions = model.get_fault_prediction(output).cpu().numpy()
            probabilities = output.cpu().detach().numpy()
            targets = target.cpu().numpy()
            
            all_predictions.extend(predictions.flatten())
            all_probabilities.extend(probabilities.flatten())
            all_targets.extend(targets.flatten())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        metrics = self.calculate_metrics(
            np.array(all_targets), 
            np.array(all_predictions), 
            np.array(all_probabilities)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate_epoch(self, 
                      model: NeuroTrustModel, 
                      val_loader: DataLoader, 
                      criterion: nn.Module) -> Dict[str, float]:
        """Validate for one epoch"""
        
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output, debug_info = model(data)
                loss = criterion(output, target)
                
                # Collect metrics
                total_loss += loss.item()
                
                # Convert to numpy for metrics calculation
                predictions = model.get_fault_prediction(output).cpu().numpy()
                probabilities = output.cpu().numpy()
                targets = target.cpu().numpy()
                
                all_predictions.extend(predictions.flatten())
                all_probabilities.extend(probabilities.flatten())
                all_targets.extend(targets.flatten())
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        metrics = self.calculate_metrics(
            np.array(all_targets), 
            np.array(all_predictions), 
            np.array(all_probabilities)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train_model(self, processed_data: Dict) -> Tuple[NeuroTrustModel, Dict[str, Any]]:
        """
        Train the NeuroTrust model on processed data
        
        Args:
            processed_data: Dictionary containing processed features and targets
            
        Returns:
            Trained model and training metrics
        """
        
        logger.info("Starting model training...")
        
        X = processed_data['features']
        y = processed_data['targets']
        
        # Create model
        model_config = {
            'input_dim': X.shape[1],
            'hidden_dim': self.config.model.hidden_dim,
            'num_gru_layers': self.config.model.num_gru_layers,
            'num_rules': self.config.model.num_rules,
            'dropout': self.config.model.dropout
        }
        
        model = create_neurotrust_model(model_config)
        model = model.to(self.device)
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            X, y, 
            batch_size=self.config.training.batch_size,
            test_size=self.config.training.validation_split
        )
        
        # Setup training components
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Learning rate scheduler - FIXED: Removed verbose parameter
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': []
        }
        
        patience_counter = 0
        max_patience = self.config.training.early_stopping_patience
        
        for epoch in range(self.config.training.max_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.training.max_epochs}")
            
            # Training
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation
            val_metrics = self.validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_metrics['loss'])
            
            # Log metrics
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_score']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
            
            # Store training history
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['train_accuracy'].append(train_metrics['accuracy'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            training_history['train_f1'].append(train_metrics['f1_score'])
            training_history['val_f1'].append(val_metrics['f1_score'])
            
            # Early stopping and best model saving
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                logger.info("New best model saved!")
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info("Loaded best model state")
        
        # Final evaluation
        final_val_metrics = self.validate_epoch(model, val_loader, criterion)
        
        # Prepare training summary
        training_metrics = {
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss,
            'final_metrics': final_val_metrics,
            'training_history': training_history,
            'model_config': model_config
        }
        
        logger.info("Training completed successfully!")
        logger.info(f"Final validation metrics: {final_val_metrics}")
        
        return model, training_metrics