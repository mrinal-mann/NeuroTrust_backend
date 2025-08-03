"""
NeuroTrust Model: GRU + RELANFIS Neuro-Symbolic Architecture - FIXED VERSION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class RELANFISLayer(nn.Module):
    """
    RELANFIS (Relational Adaptive Network Fuzzy Inference System)
    Symbolic reasoning layer for software reliability rules - FIXED VERSION
    """
    
    def __init__(self, input_dim: int, num_rules: int = 8):
        super(RELANFISLayer, self).__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        
        # Fuzzy membership parameters (Gaussian)
        self.centers = nn.Parameter(torch.randn(num_rules, input_dim))
        self.widths = nn.Parameter(torch.ones(num_rules, input_dim) * 0.5)
        
        # Rule consequence parameters - FIXED: Correct dimension
        self.consequence_weights = nn.Parameter(torch.randn(num_rules, input_dim + 1))
        
        # Rule activation weights
        self.rule_weights = nn.Parameter(torch.ones(num_rules))
        
    def membership_degree(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate Gaussian membership degrees for each rule"""
        # x: (batch_size, input_dim)
        # centers: (num_rules, input_dim)
        
        batch_size = x.size(0)
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        centers_expanded = self.centers.unsqueeze(0)  # (1, num_rules, input_dim)
        widths_expanded = self.widths.unsqueeze(0)  # (1, num_rules, input_dim)
        
        # Gaussian membership function
        diff = (x_expanded - centers_expanded) / (widths_expanded + 1e-8)
        membership = torch.exp(-0.5 * torch.sum(diff ** 2, dim=2))  # (batch_size, num_rules)
        
        return membership
    
    def rule_activation(self, membership: torch.Tensor) -> torch.Tensor:
        """Calculate rule activation strength"""
        # Apply rule weights
        activation = membership * F.softmax(self.rule_weights, dim=0)
        return activation
    
    def consequence_layer(self, x: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        """Calculate rule consequences (TSK-style) - FIXED VERSION"""
        batch_size = x.size(0)
        
        # Add bias term to input
        x_with_bias = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)
        # x_with_bias: (batch_size, input_dim + 1)
        
        # Calculate consequences for each rule
        # consequence_weights: (num_rules, input_dim + 1)
        # x_with_bias: (batch_size, input_dim + 1)
        
        # Matrix multiplication: (batch_size, input_dim + 1) @ (input_dim + 1, num_rules)
        consequences = torch.matmul(x_with_bias, self.consequence_weights.transpose(0, 1))
        # consequences: (batch_size, num_rules)
        
        return consequences
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass through RELANFIS"""
        # Calculate membership degrees
        membership = self.membership_degree(x)
        
        # Calculate rule activations
        activation = self.rule_activation(membership)
        
        # Calculate rule consequences
        consequences = self.consequence_layer(x, activation)
        
        # Weighted average of consequences
        numerator = torch.sum(activation * consequences, dim=1)
        denominator = torch.sum(activation, dim=1) + 1e-8
        output = numerator / denominator
        
        # Return debug info
        debug_info = {
            'membership': membership,
            'activation': activation,
            'consequences': consequences,
            'rule_weights': self.rule_weights
        }
        
        return output.unsqueeze(1), debug_info

class NeuroTrustModel(nn.Module):
    """
    Complete NeuroTrust Model: GRU + RELANFIS + Output Layer - FIXED VERSION
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_gru_layers: int = 2,
                 num_rules: int = 8,
                 dropout: float = 0.2):
        super(NeuroTrustModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        # GRU layers for temporal modeling
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=False
        )
        
        # RELANFIS symbolic reasoning layer - FIXED: Use hidden_dim
        self.relanfis = RELANFISLayer(hidden_dim, num_rules)
        
        # Output layers - FIXED: Correct input dimension
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through NeuroTrust model - FIXED VERSION
        
        Args:
            x: Input tensor (batch_size, input_dim) or (batch_size, sequence_length, input_dim)
        
        Returns:
            output: Reliability score (batch_size, 1)
            debug_info: Dictionary with intermediate outputs
        """
        batch_size = x.size(0)
        
        # Handle both 2D and 3D inputs
        if len(x.shape) == 2:
            # Convert to sequence format (batch_size, 1, input_dim)
            x = x.unsqueeze(1)
        
        sequence_length = x.size(1)
        
        # Reshape for processing: (batch_size * seq_len, input_dim)
        x_reshaped = x.view(-1, self.input_dim)
        
        # Input projection
        x_proj = self.input_projection(x_reshaped)  # (batch_size * seq_len, hidden_dim)
        x_proj = self.input_dropout(x_proj)
        
        # Reshape back to sequence format
        x_proj = x_proj.view(batch_size, sequence_length, self.hidden_dim)
        
        # GRU processing
        gru_output, hidden = self.gru(x_proj)  # gru_output: (batch_size, seq_len, hidden_dim)
        
        # Take the last timestep output
        last_output = gru_output[:, -1, :]  # (batch_size, hidden_dim)
        
        # RELANFIS symbolic reasoning
        symbolic_output, relanfis_debug = self.relanfis(last_output)
        
        # Combine GRU and RELANFIS outputs
        combined = torch.cat([last_output, symbolic_output], dim=1)
        
        # Final output
        reliability_score = self.output_projection(combined)
        reliability_score = torch.sigmoid(reliability_score)
        
        # Prepare debug information
        debug_info = {
            'gru_output': gru_output,
            'gru_hidden': hidden,
            'symbolic_output': symbolic_output,
            'relanfis_debug': relanfis_debug,
            'combined_features': combined
        }
        
        return reliability_score, debug_info
    
    def get_fault_prediction(self, reliability_score: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convert reliability score to fault label"""
        # fault_label = 1 if reliability_score < threshold, 0 otherwise
        fault_label = (reliability_score < threshold).float()
        return fault_label
    
    def get_model_confidence(self, reliability_score: torch.Tensor) -> torch.Tensor:
        """Calculate model confidence based on distance from decision boundary"""
        # Confidence is higher when prediction is further from 0.5
        confidence = torch.abs(reliability_score - 0.5) * 2
        return confidence

class ModelEnsemble(nn.Module):
    """
    Ensemble of NeuroTrust models for improved reliability
    """
    
    def __init__(self, models: List[NeuroTrustModel]):
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass through ensemble"""
        outputs = []
        debug_infos = []
        
        for model in self.models:
            output, debug_info = model(x)
            outputs.append(output)
            debug_infos.append(debug_info)
        
        # Average ensemble predictions
        ensemble_output = torch.stack(outputs, dim=0).mean(dim=0)
        
        # Calculate ensemble variance for uncertainty estimation
        ensemble_variance = torch.stack(outputs, dim=0).var(dim=0)
        
        ensemble_debug = {
            'individual_outputs': outputs,
            'individual_debug': debug_infos,
            'ensemble_variance': ensemble_variance
        }
        
        return ensemble_output, ensemble_debug

# Factory function for creating models
def create_neurotrust_model(config: Dict) -> NeuroTrustModel:
    """Create NeuroTrust model from configuration"""
    model = NeuroTrustModel(
        input_dim=config.get('input_dim', 5),
        hidden_dim=config.get('hidden_dim', 64),
        num_gru_layers=config.get('num_gru_layers', 2),
        num_rules=config.get('num_rules', 8),
        dropout=config.get('dropout', 0.2)
    )
    
    logger.info(f"Created NeuroTrust model:")
    logger.info(f"  Input dim: {config.get('input_dim', 5)}")
    logger.info(f"  Hidden dim: {config.get('hidden_dim', 64)}")
    logger.info(f"  GRU layers: {config.get('num_gru_layers', 2)}")
    logger.info(f"  RELANFIS rules: {config.get('num_rules', 8)}")
    logger.info(f"  Dropout: {config.get('dropout', 0.2)}")
    
    return model