"""
NeuroTrust Model Predictor
Handles inference and SHAP explanations for trained models
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging
import shap
from sklearn.preprocessing import LabelEncoder

from models.neurotrust_model import NeuroTrustModel, create_neurotrust_model

logger = logging.getLogger(__name__)

class ModelPredictor:
    """Predictor class for NeuroTrust model with SHAP explanations"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_columns = None
        self.scaler_params = None
        self.label_encoders = {}
        
    def load_model(self, model_path: str = "model/model.pt"):
        """Load trained model and preprocessing parameters"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration
            model_config = {
                'input_dim': len(checkpoint['feature_columns']),
                'hidden_dim': self.config.model.hidden_dim,
                'num_gru_layers': self.config.model.num_gru_layers,
                'num_rules': self.config.model.num_rules,
                'dropout': 0.0  # No dropout during inference
            }
            
            # Create and load model
            self.model = create_neurotrust_model(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load preprocessing parameters
            self.feature_columns = checkpoint['feature_columns']
            self.scaler_params = checkpoint['scaler_params']
            
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Feature columns: {self.feature_columns}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess input data to match training format"""
        
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Handle categorical variables
        if 'last_editor_experience' in df.columns:
            # Map experience levels to numeric values
            experience_map = {'junior': 0, 'mid': 1, 'senior': 2}
            df['last_editor_experience'] = df['last_editor_experience'].map(experience_map)
        
        # Ensure all expected features are present
        for col in self.feature_columns:
            if col not in df.columns:
                logger.warning(f"Missing feature {col}, setting to 0")
                df[col] = 0
        
        # Select and order features
        X = df[self.feature_columns].values.astype(np.float32)
        
        # Apply scaling if available
        if self.scaler_params:
            mean = np.array(self.scaler_params['mean'])
            std = np.array(self.scaler_params['std'])
            X = (X - mean) / (std + 1e-8)
        
        return X
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction and generate SHAP explanations
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Dictionary with prediction results and explanations
        """
        
        if self.model is None:
            self.load_model()
        
        # Preprocess input
        X = self.preprocess_input(input_data)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            reliability_score, debug_info = self.model(X_tensor)
            fault_label = self.model.get_fault_prediction(reliability_score)
            confidence = self.model.get_model_confidence(reliability_score)
        
        # Convert to Python types
        reliability_score = float(reliability_score.cpu().numpy()[0, 0])
        fault_label = int(fault_label.cpu().numpy()[0, 0])
        confidence = float(confidence.cpu().numpy()[0, 0])
        
        # Generate SHAP explanations
        shap_values = self.explain_prediction(X)
        
        # Prepare result
        result = {
            'fault_label': fault_label,
            'reliability_score': reliability_score,
            'model_confidence': confidence,
            'shap_values': shap_values
        }
        
        logger.info(f"Prediction: fault_label={fault_label}, "
                   f"reliability_score={reliability_score:.4f}, "
                   f"confidence={confidence:.4f}")
        
        return result
    
    def explain_prediction(self, X: np.ndarray, num_background_samples: int = 100) -> Dict[str, float]:
        """
        Generate SHAP explanations for the prediction
        
        Args:
            X: Input features (1 sample)
            num_background_samples: Number of background samples for SHAP
            
        Returns:
            Dictionary with SHAP values for each feature
        """
        
        try:
            # Create background dataset for SHAP
            # Generate synthetic background data
            background_data = self.generate_background_data(num_background_samples)
            
            # Define prediction function for SHAP
            def predict_fn(x):
                """Prediction function for SHAP"""
                x_tensor = torch.FloatTensor(x).to(self.device)
                with torch.no_grad():
                    output, _ = self.model(x_tensor)
                return output.cpu().numpy()
            
            # Create SHAP explainer
            explainer = shap.KernelExplainer(predict_fn, background_data)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X, nsamples=50)
            
            # Create feature-wise SHAP dictionary
            shap_dict = {}
            for i, feature_name in enumerate(self.feature_columns):
                shap_dict[feature_name] = float(shap_values[0, i])
            
            # Sort by absolute importance
            shap_dict = dict(sorted(shap_dict.items(), 
                                  key=lambda x: abs(x[1]), 
                                  reverse=True))
            
            logger.info("SHAP explanations generated successfully")
            
            return shap_dict
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {str(e)}")
            # Return default explanations
            return {feature: 0.0 for feature in self.feature_columns}
    
    def generate_background_data(self, num_samples: int = 100) -> np.ndarray:
        """
        Generate background data for SHAP explanations
        
        Args:
            num_samples: Number of background samples to generate
            
        Returns:
            Background data array
        """
        
        # Generate reasonable ranges for each feature
        feature_ranges = {
            'file_complexity': (1, 50),
            'recent_commits': (0, 20),
            'defects': (0, 10),
            'lines_of_code': (10, 5000),
            'last_editor_experience': (0, 2)  # 0=junior, 1=mid, 2=senior
        }
        
        background_data = []
        
        for _ in range(num_samples):
            sample = []
            for feature in self.feature_columns:
                if feature in feature_ranges:
                    min_val, max_val = feature_ranges[feature]
                    if feature == 'last_editor_experience':
                        # Discrete values for experience
                        value = np.random.choice([0, 1, 2])
                    else:
                        value = np.random.uniform(min_val, max_val)
                else:
                    # Default range for unknown features
                    value = np.random.uniform(0, 1)
                
                sample.append(value)
            
            background_data.append(sample)
        
        background_data = np.array(background_data, dtype=np.float32)
        
        # Apply same scaling as training data
        if self.scaler_params:
            mean = np.array(self.scaler_params['mean'])
            std = np.array(self.scaler_params['std'])
            background_data = (background_data - mean) / (std + 1e-8)
        
        return background_data
    
    def batch_predict(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple inputs
        
        Args:
            input_list: List of input dictionaries
            
        Returns:
            List of prediction results
        """
        
        if self.model is None:
            self.load_model()
        
        results = []
        
        for input_data in input_list:
            try:
                result = self.predict(input_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for input {input_data}: {str(e)}")
                results.append({
                    'fault_label': -1,
                    'reliability_score': 0.0,
                    'model_confidence': 0.0,
                    'shap_values': {},
                    'error': str(e)
                })
        
        return results
    
    def get_feature_importance(self, num_samples: int = 200) -> Dict[str, float]:
        """
        Get global feature importance using SHAP
        
        Args:
            num_samples: Number of samples for importance calculation
            
        Returns:
            Dictionary with global feature importance
        """
        
        if self.model is None:
            self.load_model()
        
        try:
            # Generate diverse test data
            test_data = self.generate_background_data(num_samples)
            
            # Define prediction function
            def predict_fn(x):
                x_tensor = torch.FloatTensor(x).to(self.device)
                with torch.no_grad():
                    output, _ = self.model(x_tensor)
                return output.cpu().numpy()
            
            # Create SHAP explainer with smaller background
            background_data = self.generate_background_data(50)
            explainer = shap.KernelExplainer(predict_fn, background_data)
            
            # Calculate SHAP values for test data
            shap_values = explainer.shap_values(test_data[:50], nsamples=30)  # Reduced for speed
            
            # Calculate mean absolute SHAP values (global importance)
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create feature importance dictionary
            importance_dict = {}
            for i, feature_name in enumerate(self.feature_columns):
                importance_dict[feature_name] = float(mean_shap[i])
            
            # Normalize to sum to 1
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {k: v / total_importance 
                                 for k, v in importance_dict.items()}
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True))
            
            logger.info("Global feature importance calculated successfully")
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {feature: 1.0 / len(self.feature_columns) 
                   for feature in self.feature_columns}