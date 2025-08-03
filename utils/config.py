"""
NeuroTrust Configuration Module
Central configuration management for the entire system
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    hidden_dim: int = 64
    num_gru_layers: int = 2
    num_rules: int = 8
    dropout: float = 0.2
    input_dim: int = 5  # Will be set dynamically based on data

@dataclass
class TrainingConfig:
    """Training hyperparameters configuration"""
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    max_epochs: int = 50
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    gradient_clip_norm: float = 1.0

@dataclass
class FederatedConfig:
    """Federated learning configuration"""
    server_address: str = "localhost:8080"
    num_rounds: int = 3
    min_clients: int = 1
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0

@dataclass
class DataConfig:
    """Data processing configuration"""
    missing_value_strategy: str = "median"  # median, mean, mode
    feature_scaling: str = "standard"  # standard, minmax, robust
    categorical_encoding: str = "label"  # label, onehot
    remove_outliers: bool = False
    outlier_method: str = "iqr"  # iqr, zscore

@dataclass
class ShapConfig:
    """SHAP explainability configuration"""
    background_samples: int = 100
    explanation_samples: int = 50
    kernel_shap_samples: int = 30
    feature_importance_samples: int = 200

class Config:
    """Main configuration class"""
    
    def __init__(self):
        # Initialize all configuration sections
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.federated = FederatedConfig()
        self.data = DataConfig()
        self.shap = ShapConfig()
        
        # System configuration
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.data_dir = os.getenv("DATA_DIR", "data")
        self.model_dir = os.getenv("MODEL_DIR", "model")
        
        # API configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        
        # Load from environment variables if available
        self._load_from_env()
        
        logger.info("Configuration initialized")
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        
        # Model configuration
        if os.getenv("MODEL_HIDDEN_DIM"):
            self.model.hidden_dim = int(os.getenv("MODEL_HIDDEN_DIM"))
        
        if os.getenv("MODEL_GRU_LAYERS"):
            self.model.num_gru_layers = int(os.getenv("MODEL_GRU_LAYERS"))
        
        if os.getenv("MODEL_NUM_RULES"):
            self.model.num_rules = int(os.getenv("MODEL_NUM_RULES"))
        
        if os.getenv("MODEL_DROPOUT"):
            self.model.dropout = float(os.getenv("MODEL_DROPOUT"))
        
        # Training configuration
        if os.getenv("TRAINING_BATCH_SIZE"):
            self.training.batch_size = int(os.getenv("TRAINING_BATCH_SIZE"))
        
        if os.getenv("TRAINING_LR"):
            self.training.learning_rate = float(os.getenv("TRAINING_LR"))
        
        if os.getenv("TRAINING_MAX_EPOCHS"):
            self.training.max_epochs = int(os.getenv("TRAINING_MAX_EPOCHS"))
        
        if os.getenv("TRAINING_PATIENCE"):
            self.training.early_stopping_patience = int(os.getenv("TRAINING_PATIENCE"))
        
        # Federated learning configuration
        if os.getenv("FL_SERVER_ADDRESS"):
            self.federated.server_address = os.getenv("FL_SERVER_ADDRESS")
        
        if os.getenv("FL_NUM_ROUNDS"):
            self.federated.num_rounds = int(os.getenv("FL_NUM_ROUNDS"))
        
        # SHAP configuration
        if os.getenv("SHAP_BACKGROUND_SAMPLES"):
            self.shap.background_samples = int(os.getenv("SHAP_BACKGROUND_SAMPLES"))
        
        if os.getenv("SHAP_EXPLANATION_SAMPLES"):
            self.shap.explanation_samples = int(os.getenv("SHAP_EXPLANATION_SAMPLES"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        
        return {
            "model": {
                "hidden_dim": self.model.hidden_dim,
                "num_gru_layers": self.model.num_gru_layers,
                "num_rules": self.model.num_rules,
                "dropout": self.model.dropout,
                "input_dim": self.model.input_dim
            },
            "training": {
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "max_epochs": self.training.max_epochs,
                "early_stopping_patience": self.training.early_stopping_patience,
                "validation_split": self.training.validation_split,
                "gradient_clip_norm": self.training.gradient_clip_norm
            },
            "federated": {
                "server_address": self.federated.server_address,
                "num_rounds": self.federated.num_rounds,
                "min_clients": self.federated.min_clients,
                "fraction_fit": self.federated.fraction_fit,
                "fraction_evaluate": self.federated.fraction_evaluate
            },
            "data": {
                "missing_value_strategy": self.data.missing_value_strategy,
                "feature_scaling": self.data.feature_scaling,
                "categorical_encoding": self.data.categorical_encoding,
                "remove_outliers": self.data.remove_outliers,
                "outlier_method": self.data.outlier_method
            },
            "shap": {
                "background_samples": self.shap.background_samples,
                "explanation_samples": self.shap.explanation_samples,
                "kernel_shap_samples": self.shap.kernel_shap_samples,
                "feature_importance_samples": self.shap.feature_importance_samples
            },
            "system": {
                "debug": self.debug,
                "log_level": self.log_level,
                "data_dir": self.data_dir,
                "model_dir": self.model_dir,
                "api_host": self.api_host,
                "api_port": self.api_port,
                "cors_origins": self.cors_origins
            }
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        
        if "model" in config_dict:
            model_config = config_dict["model"]
            self.model.hidden_dim = model_config.get("hidden_dim", self.model.hidden_dim)
            self.model.num_gru_layers = model_config.get("num_gru_layers", self.model.num_gru_layers)
            self.model.num_rules = model_config.get("num_rules", self.model.num_rules)
            self.model.dropout = model_config.get("dropout", self.model.dropout)
            self.model.input_dim = model_config.get("input_dim", self.model.input_dim)
        
        if "training" in config_dict:
            training_config = config_dict["training"]
            self.training.batch_size = training_config.get("batch_size", self.training.batch_size)
            self.training.learning_rate = training_config.get("learning_rate", self.training.learning_rate)
            self.training.max_epochs = training_config.get("max_epochs", self.training.max_epochs)
            self.training.early_stopping_patience = training_config.get("early_stopping_patience", self.training.early_stopping_patience)
        
        # Update other sections as needed
        logger.info("Configuration updated from dictionary")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        
        errors = []
        
        # Validate model configuration
        if self.model.hidden_dim <= 0:
            errors.append("Model hidden_dim must be positive")
        
        if self.model.num_gru_layers <= 0:
            errors.append("Model num_gru_layers must be positive")
        
        if self.model.num_rules <= 0:
            errors.append("Model num_rules must be positive")
        
        if not 0 <= self.model.dropout < 1:
            errors.append("Model dropout must be between 0 and 1")
        
        # Validate training configuration
        if self.training.batch_size <= 0:
            errors.append("Training batch_size must be positive")
        
        if self.training.learning_rate <= 0:
            errors.append("Training learning_rate must be positive")
        
        if self.training.max_epochs <= 0:
            errors.append("Training max_epochs must be positive")
        
        if not 0 < self.training.validation_split < 1:
            errors.append("Training validation_split must be between 0 and 1")
        
        # Validate federated configuration
        if self.federated.num_rounds <= 0:
            errors.append("Federated num_rounds must be positive")
        
        if not 0 < self.federated.fraction_fit <= 1:
            errors.append("Federated fraction_fit must be between 0 and 1")
        
        # Validate SHAP configuration
        if self.shap.background_samples <= 0:
            errors.append("SHAP background_samples must be positive")
        
        if self.shap.explanation_samples <= 0:
            errors.append("SHAP explanation_samples must be positive")
        
        return errors
    
    def get_model_config_for_creation(self, input_dim: int) -> Dict[str, Any]:
        """Get model configuration dictionary for model creation"""
        
        return {
            'input_dim': input_dim,
            'hidden_dim': self.model.hidden_dim,
            'num_gru_layers': self.model.num_gru_layers,
            'num_rules': self.model.num_rules,
            'dropout': self.model.dropout
        }
    
    def __str__(self) -> str:
        """String representation of configuration"""
        
        return f"""NeuroTrust Configuration:
Model: hidden_dim={self.model.hidden_dim}, gru_layers={self.model.num_gru_layers}, rules={self.model.num_rules}
Training: batch_size={self.training.batch_size}, lr={self.training.learning_rate}, epochs={self.training.max_epochs}
Federated: server={self.federated.server_address}, rounds={self.federated.num_rounds}
SHAP: background={self.shap.background_samples}, explanation={self.shap.explanation_samples}
System: debug={self.debug}, api_port={self.api_port}"""

# Global configuration instance
config = Config()

# Configuration validation
def validate_config():
    """Validate global configuration and log any errors"""
    
    errors = config.validate()
    if errors:
        logger.error("Configuration validation errors:")
        for error in errors:
            logger.error(f"  - {error}")
        raise ValueError(f"Invalid configuration: {errors}")
    else:
        logger.info("Configuration validation passed")

# Environment-specific configurations
def get_development_config() -> Config:
    """Get configuration optimized for development"""
    
    dev_config = Config()
    dev_config.debug = True
    dev_config.training.max_epochs = 10  # Faster training for development
    dev_config.training.early_stopping_patience = 3
    dev_config.shap.background_samples = 50  # Faster SHAP for development
    dev_config.shap.explanation_samples = 20
    
    return dev_config

def get_production_config() -> Config:
    """Get configuration optimized for production"""
    
    prod_config = Config()
    prod_config.debug = False
    prod_config.training.max_epochs = 100
    prod_config.training.early_stopping_patience = 15
    prod_config.shap.background_samples = 200
    prod_config.shap.explanation_samples = 100
    
    return prod_config

def get_config_for_environment(env: str = None) -> Config:
    """Get configuration based on environment"""
    
    if env is None:
        env = os.getenv("ENVIRONMENT", "development")
    
    if env.lower() == "production":
        return get_production_config()
    else:
        return get_development_config()