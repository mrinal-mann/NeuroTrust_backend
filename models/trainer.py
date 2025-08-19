"""
Enhanced NeuroTrust Trainer with Beautiful Progress Bars and Animations
Add this to your existing trainer.py or create as a new enhanced version
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Tuple, Any, List
import logging
import time
import sys

# Progress bar imports
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for Windows
colorama.init()

from models.neurotrust_model import NeuroTrustModel, create_neurotrust_model

logger = logging.getLogger(__name__)

class EnhancedEnsembleTrainer:
    """Enhanced Ensemble trainer with beautiful progress bars and animations"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.baseline_to_beat = 0.791  # Random Forest baseline
        
        # Progress tracking
        self.current_model = 0
        self.total_models = 5
        self.model_progress = {}
    
    def print_fancy_header(self):
        """Print beautiful training header"""
        print("\n" + "="*80)
        print(f"{Fore.CYAN}🚀 NEUROTRUST ENSEMBLE TRAINING - MEGA EDITION 🚀{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🎯 TARGET: Beat Random Forest {self.baseline_to_beat:.1%} baseline{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📊 Training 5 diverse neural networks on MASSIVE dataset{Style.RESET_ALL}")
        print("="*80)
    
    def create_diverse_model_configs(self) -> List[Dict[str, Any]]:
        """Create diverse model configurations for ensemble"""
        
        configs = [
            # Model 1: Deep and narrow - good for complex patterns
            {
                'name': 'DeepNarrow',
                'emoji': '🏗️',
                'description': 'Deep & Narrow - Complex Pattern Detection',
                'input_dim': 21,
                'hidden_dim': 64,
                'num_gru_layers': 3,
                'num_rules': 8,
                'dropout': 0.3
            },
            # Model 2: Wide and shallow - good for feature interactions
            {
                'name': 'WideShallow',
                'emoji': '🏭',
                'description': 'Wide & Shallow - Feature Interaction Expert',
                'input_dim': 21,
                'hidden_dim': 128,
                'num_gru_layers': 1,
                'num_rules': 16,
                'dropout': 0.4
            },
            # Model 3: Balanced - general purpose
            {
                'name': 'Balanced',
                'emoji': '⚖️',
                'description': 'Balanced Architecture - General Purpose',
                'input_dim': 21,
                'hidden_dim': 96,
                'num_gru_layers': 2,
                'num_rules': 12,
                'dropout': 0.35
            },
            # Model 4: High capacity - for difficult patterns
            {
                'name': 'HighCapacity',
                'emoji': '💪',
                'description': 'High Capacity - Difficult Pattern Specialist',
                'input_dim': 21,
                'hidden_dim': 160,
                'num_gru_layers': 2,
                'num_rules': 20,
                'dropout': 0.5
            },
            # Model 5: Conservative - stable predictions
            {
                'name': 'Conservative',
                'emoji': '🛡️',
                'description': 'Conservative - Stable & Reliable',
                'input_dim': 21,
                'hidden_dim': 48,
                'num_gru_layers': 2,
                'num_rules': 6,
                'dropout': 0.2
            }
        ]
        
        return configs
    
    def create_ensemble_data_loaders(self, X: np.ndarray, y: np.ndarray) -> List[Tuple]:
        """Create different data loader configurations for diversity"""
        
        print(f"\n{Fore.BLUE}📊 Creating diverse data configurations...{Style.RESET_ALL}")
        
        loaders = []
        
        # Show data preparation progress
        with tqdm(total=5, desc="🔄 Preparing data splits", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                 colour='blue') as pbar:
            
            # Configuration 1: Aggressive minority boosting
            X_train1, X_val1, y_train1, y_val1 = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            class_weights1 = compute_class_weight('balanced', classes=np.unique(y_train1), y=y_train1)
            class_weights1[1] *= 2.5
            sample_weights1 = np.array([class_weights1[int(label)] for label in y_train1])
            sampler1 = WeightedRandomSampler(sample_weights1, int(len(sample_weights1) * 1.8), replacement=True)
            pbar.update(1)
            
            # Configuration 2: Moderate minority boosting
            X_train2, X_val2, y_train2, y_val2 = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
            class_weights2 = compute_class_weight('balanced', classes=np.unique(y_train2), y=y_train2)
            class_weights2[1] *= 1.8
            sample_weights2 = np.array([class_weights2[int(label)] for label in y_train2])
            sampler2 = WeightedRandomSampler(sample_weights2, int(len(sample_weights2) * 1.4), replacement=True)
            pbar.update(1)
            
            # Configuration 3: Conservative boosting
            X_train3, X_val3, y_train3, y_val3 = train_test_split(X, y, test_size=0.2, random_state=456, stratify=y)
            class_weights3 = compute_class_weight('balanced', classes=np.unique(y_train3), y=y_train3)
            class_weights3[1] *= 1.3
            sample_weights3 = np.array([class_weights3[int(label)] for label in y_train3])
            sampler3 = WeightedRandomSampler(sample_weights3, int(len(sample_weights3) * 1.2), replacement=True)
            pbar.update(1)
            
            # Configuration 4: Focal loss approach
            X_train4, X_val4, y_train4, y_val4 = train_test_split(X, y, test_size=0.25, random_state=789, stratify=y)
            class_weights4 = compute_class_weight('balanced', classes=np.unique(y_train4), y=y_train4)
            class_weights4[1] *= 2.0
            sample_weights4 = np.array([class_weights4[int(label)] for label in y_train4])
            sampler4 = WeightedRandomSampler(sample_weights4, int(len(sample_weights4) * 1.6), replacement=True)
            pbar.update(1)
            
            # Configuration 5: Standard approach
            X_train5, X_val5, y_train5, y_val5 = train_test_split(X, y, test_size=0.25, random_state=999, stratify=y)
            class_weights5 = compute_class_weight('balanced', classes=np.unique(y_train5), y=y_train5)
            class_weights5[1] *= 1.5
            sample_weights5 = np.array([class_weights5[int(label)] for label in y_train5])
            sampler5 = WeightedRandomSampler(sample_weights5, int(len(sample_weights5) * 1.3), replacement=True)
            pbar.update(1)
        
        # Create data loaders with progress
        print(f"{Fore.GREEN}✅ Data configurations created successfully!{Style.RESET_ALL}")
        
        for i, (X_train, X_val, y_train, y_val, sampler, class_weights) in enumerate([
            (X_train1, X_val1, y_train1, y_val1, sampler1, class_weights1),
            (X_train2, X_val2, y_train2, y_val2, sampler2, class_weights2),
            (X_train3, X_val3, y_train3, y_val3, sampler3, class_weights3),
            (X_train4, X_val4, y_train4, y_val4, sampler4, class_weights4),
            (X_train5, X_val5, y_train5, y_val5, sampler5, class_weights5)
        ]):
            
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            batch_size = 64 if i < 3 else 48
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
            val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
            
            loaders.append((train_loader, val_loader, class_weights, f"Config_{i+1}"))
        
        return loaders
    
    def train_single_model_with_progress(self, model_config: Dict, data_config: Tuple, model_index: int) -> Tuple[NeuroTrustModel, Dict]:
        """Train a single model with beautiful progress bars"""
        
        train_loader, val_loader, class_weights, config_name = data_config
        
        # Model header
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{model_config['emoji']} MODEL {model_index+1}: {model_config['name']}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{model_config['description']}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Create model
        model = create_neurotrust_model(model_config)
        model = model.to(self.device)
        
        # Optimizer setup
        learning_rates = [0.0005, 0.001, 0.0015, 0.0008, 0.0012]
        lr = learning_rates[model_index % len(learning_rates)]
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
        
        # Training parameters
        max_epochs = 100
        best_score = 0.0
        best_model = None
        patience = 0
        max_patience = 20
        
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        # Training progress bar
        epoch_pbar = tqdm(range(max_epochs), 
                         desc=f"🚀 Training {model_config['name']}", 
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                         colour='green')
        
        for epoch in epoch_pbar:
            # Training phase
            model.train()
            train_loss = 0.0
            batch_count = 0
            
            # Batch progress (nested progress bar)
            batch_pbar = tqdm(train_loader, 
                            desc=f"  📦 Epoch {epoch+1} Batches", 
                            leave=False,
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                            colour='blue')
            
            for data, target in batch_pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output, _ = model(data)
                
                # Advanced weighted loss with focal component
                bce_loss = nn.BCELoss(reduction='none')(output, target)
                weights = torch.where(target == 1, class_weights_tensor[1], class_weights_tensor[0])
                
                # Add focal loss component
                pt = torch.where(target == 1, output, 1 - output)
                focal_weight = (1 - pt) ** 1.5
                
                weighted_loss = (bce_loss * weights.unsqueeze(1) * focal_weight).mean()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += weighted_loss.item()
                batch_count += 1
                
                # Update batch progress
                batch_pbar.set_postfix({'Loss': f"{weighted_loss.item():.4f}"})
            
            scheduler.step()
            train_loss /= batch_count
            
            # Validation phase
            model.eval()
            all_probs = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output, _ = model(data)
                    all_probs.extend(output.cpu().numpy().flatten())
                    all_targets.extend(target.cpu().numpy().flatten())
            
            # Calculate metrics
            all_probs = np.array(all_probs)
            all_targets = np.array(all_targets)
            all_preds = (all_probs >= 0.5).astype(int)
            
            accuracy = accuracy_score(all_targets, all_preds)
            recall = recall_score(all_targets, all_preds, zero_division=0)
            f1 = f1_score(all_targets, all_preds, zero_division=0)
            
            # Score calculation
            score = accuracy * 0.7 + recall * 0.2 + f1 * 0.1
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'Acc': f"{accuracy:.3f}",
                'Rec': f"{recall:.3f}", 
                'F1': f"{f1:.3f}",
                'Loss': f"{train_loss:.4f}",
                'Best': f"{best_score:.3f}"
            })
            
            # Best model tracking
            if score > best_score:
                best_score = score
                best_model = model.state_dict().copy()
                patience = 0
                
                # Celebration for beating baseline
                if accuracy > self.baseline_to_beat:
                    epoch_pbar.write(f"  🎉 {Fore.GREEN}BEATS BASELINE! Acc: {accuracy:.3f} > {self.baseline_to_beat:.3f}{Style.RESET_ALL}")
            else:
                patience += 1
            
            # Early stopping
            if patience >= max_patience:
                epoch_pbar.write(f"  ⏹️ Early stopping at epoch {epoch+1}")
                break
        
        epoch_pbar.close()
        
        # Load best model
        if best_model is not None:
            model.load_state_dict(best_model)
        
        # Final validation
        model.eval()
        final_probs = []
        final_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = model(data)
                final_probs.extend(output.cpu().numpy().flatten())
                final_targets.extend(target.cpu().numpy().flatten())
        
        final_metrics = {
            'accuracy': accuracy_score(final_targets, (np.array(final_probs) >= 0.5).astype(int)),
            'recall': recall_score(final_targets, (np.array(final_probs) >= 0.5).astype(int), zero_division=0),
            'f1_score': f1_score(final_targets, (np.array(final_probs) >= 0.5).astype(int), zero_division=0),
            'auc': roc_auc_score(final_targets, final_probs) if len(np.unique(final_targets)) > 1 else 0.0
        }
        
        # Results display
        print(f"\n{Fore.GREEN}✅ {model_config['emoji']} MODEL {model_index+1} COMPLETED!{Style.RESET_ALL}")
        print(f"   {Fore.YELLOW}Accuracy: {final_metrics['accuracy']:.1%}{Style.RESET_ALL}")
        print(f"   {Fore.YELLOW}Recall:   {final_metrics['recall']:.1%}{Style.RESET_ALL}")
        print(f"   {Fore.YELLOW}F1-Score: {final_metrics['f1_score']:.1%}{Style.RESET_ALL}")
        print(f"   {Fore.YELLOW}AUC:      {final_metrics['auc']:.3f}{Style.RESET_ALL}")
        
        return model, {
            'model_config': model_config,
            'final_metrics': final_metrics,
            'epochs_trained': epoch + 1,
            'best_score': best_score
        }
    
    def optimize_ensemble_threshold_with_progress(self, models: List[NeuroTrustModel], X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, Dict]:
        """Optimize threshold for ensemble predictions with progress"""
        
        print(f"\n{Fore.MAGENTA}🎯 ENSEMBLE THRESHOLD OPTIMIZATION{Style.RESET_ALL}")
        
        # Get ensemble predictions with progress
        ensemble_probs = []
        
        with tqdm(models, desc="🔮 Generating ensemble predictions", colour='magenta') as model_pbar:
            for model in model_pbar:
                model.eval()
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                
                with torch.no_grad():
                    output, _ = model(X_val_tensor)
                    probs = output.cpu().numpy().flatten()
                    ensemble_probs.append(probs)
        
        # Average ensemble predictions
        avg_probs = np.mean(ensemble_probs, axis=0)
        
        # Find optimal threshold with progress
        best_threshold = 0.5
        best_accuracy = 0.0
        best_metrics = {}
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        
        with tqdm(thresholds, desc="🎚️ Optimizing threshold", colour='cyan') as thresh_pbar:
            for threshold in thresh_pbar:
                preds = (avg_probs >= threshold).astype(int)
                accuracy = accuracy_score(y_val, preds)
                
                thresh_pbar.set_postfix({'Threshold': f"{threshold:.2f}", 'Acc': f"{accuracy:.3f}"})
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
                    best_metrics = {
                        'threshold': threshold,
                        'accuracy': accuracy,
                        'recall': recall_score(y_val, preds, zero_division=0),
                        'f1_score': f1_score(y_val, preds, zero_division=0),
                        'precision': precision_score(y_val, preds, zero_division=0)
                    }
        
        print(f"\n{Fore.GREEN}🎯 OPTIMAL THRESHOLD FOUND: {best_threshold:.3f}{Style.RESET_ALL}")
        print(f"   {Fore.YELLOW}Ensemble Accuracy: {best_metrics['accuracy']:.1%}{Style.RESET_ALL}")
        
        return best_threshold, best_metrics
    
    def train_model(self, processed_data: Dict) -> Tuple[NeuroTrustModel, Dict[str, Any]]:
        """Train ensemble of models with beautiful progress animations"""
        
        # Beautiful header
        self.print_fancy_header()
        
        X = processed_data['features']
        y = processed_data['targets']
        
        print(f"\n{Fore.BLUE}📊 Dataset Statistics:{Style.RESET_ALL}")
        print(f"   Samples: {len(X):,}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Class 0: {sum(y == 0):,}")
        print(f"   Class 1: {sum(y == 1):,}")
        print(f"   Balance Ratio: {sum(y == 0) / sum(y == 1):.1f}:1")
        
        # Create model configurations
        model_configs = self.create_diverse_model_configs()
        
        # Create data configurations
        data_configs = self.create_ensemble_data_loaders(X, y)
        
        # Overall progress
        overall_pbar = tqdm(total=len(model_configs), 
                           desc="🏆 Training Ensemble", 
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} Models [{elapsed}<{remaining}]",
                           colour='red')
        
        # Train individual models
        ensemble_models = []
        individual_results = []
        
        for i, (model_config, data_config) in enumerate(zip(model_configs, data_configs)):
            model, results = self.train_single_model_with_progress(model_config, data_config, i)
            ensemble_models.append(model)
            individual_results.append(results)
            overall_pbar.update(1)
        
        overall_pbar.close()
        
        # Create final validation set
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Optimize ensemble threshold
        optimal_threshold, ensemble_metrics = self.optimize_ensemble_threshold_with_progress(
            ensemble_models, X_val_final, y_val_final
        )
        
        # Set threshold in all models
        for model in ensemble_models:
            model.optimal_threshold = optimal_threshold
        
        # Select best single model as primary
        best_model_idx = max(range(len(individual_results)), 
                           key=lambda i: individual_results[i]['final_metrics']['accuracy'])
        best_model = ensemble_models[best_model_idx]
        
        # Display final results
        self.display_final_results(individual_results, ensemble_metrics, model_configs)
        
        # Prepare comprehensive results
        training_metrics = {
            'epochs_trained': max(r['epochs_trained'] for r in individual_results),
            'best_val_loss': 0.0,
            'optimal_threshold': optimal_threshold,
            'final_metrics': ensemble_metrics,
            'individual_results': individual_results,
            'ensemble_size': len(ensemble_models),
            'baseline_to_beat': self.baseline_to_beat,
            'beats_baseline': ensemble_metrics['accuracy'] > self.baseline_to_beat,
            'model_config': model_configs[best_model_idx],
            'training_type': 'enhanced_ensemble'
        }
        
        return best_model, training_metrics
    
    def display_final_results(self, individual_results, ensemble_metrics, model_configs):
        """Display beautiful final results"""
        
        print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🏁 ENSEMBLE TRAINING COMPLETED! 🏁{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
        
        print(f"\n{Fore.YELLOW}🏆 INDIVIDUAL MODEL RESULTS:{Style.RESET_ALL}")
        for i, (config, results) in enumerate(zip(model_configs, individual_results)):
            acc = results['final_metrics']['accuracy']
            emoji = "🎉" if acc > self.baseline_to_beat else "📈"
            status = "BEATS BASELINE" if acc > self.baseline_to_beat else "Below baseline"
            print(f"   {config['emoji']} Model {i+1} ({config['name']}): {acc:.1%} - {emoji} {status}")
        
        print(f"\n{Fore.CYAN}🎯 ENSEMBLE RESULTS:{Style.RESET_ALL}")
        print(f"   {Fore.GREEN}✅ Accuracy:  {ensemble_metrics['accuracy']:.1%}{Style.RESET_ALL}")
        print(f"   {Fore.GREEN}✅ Recall:    {ensemble_metrics['recall']:.1%}{Style.RESET_ALL}")
        print(f"   {Fore.GREEN}✅ F1-Score:  {ensemble_metrics['f1_score']:.1%}{Style.RESET_ALL}")
        print(f"   {Fore.GREEN}✅ Precision: {ensemble_metrics['precision']:.1%}{Style.RESET_ALL}")
        print(f"   {Fore.GREEN}✅ Threshold: {ensemble_metrics['threshold']:.3f}{Style.RESET_ALL}")
        
        print(f"\n{Fore.MAGENTA}📊 BASELINE COMPARISON:{Style.RESET_ALL}")
        print(f"   Random Forest Baseline: {self.baseline_to_beat:.1%}")
        print(f"   Our Ensemble:          {ensemble_metrics['accuracy']:.1%}")
        
        if ensemble_metrics['accuracy'] > self.baseline_to_beat:
            improvement = ((ensemble_metrics['accuracy'] / self.baseline_to_beat) - 1) * 100
            print(f"   {Fore.GREEN}🎉🎉🎉 SUCCESS: +{improvement:.1f}% IMPROVEMENT! 🎉🎉🎉{Style.RESET_ALL}")
        else:
            gap = (self.baseline_to_beat - ensemble_metrics['accuracy']) * 100
            print(f"   {Fore.RED}📈 Gap: -{gap:.1f}% (need improvement){Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")

# Export the enhanced trainer
ModelTrainer = EnhancedEnsembleTrainer