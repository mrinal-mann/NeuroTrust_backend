"""
NeuroTrust Model Predictor - UPDATED FOR REAL DATASETS
Handles inference and SHAP explanations for software metrics datasets
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging
import shap
from sklearn.preprocessing import StandardScaler

from models.neurotrust_model import NeuroTrustModel, create_neurotrust_model

logger = logging.getLogger(__name__)

class ModelPredictor:
    """Predictor class for NeuroTrust model with real software metrics"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_columns = None
        self.scaler_params = None
        
        # Standard 21 software metrics features
        self.standard_features = [
            "LOC_BLANK", "BRANCH_COUNT", "LOC_CODE_AND_COMMENT", "LOC_COMMENTS",
            "CYCLOMATIC_COMPLEXITY", "DESIGN_COMPLEXITY", "ESSENTIAL_COMPLEXITY", 
            "LOC_EXECUTABLE", "HALSTEAD_CONTENT", "HALSTEAD_DIFFICULTY", 
            "HALSTEAD_EFFORT", "HALSTEAD_ERROR_EST", "HALSTEAD_LENGTH", 
            "HALSTEAD_LEVEL", "HALSTEAD_PROG_TIME", "HALSTEAD_VOLUME",
            "NUM_OPERANDS", "NUM_OPERATORS", "NUM_UNIQUE_OPERANDS", 
            "NUM_UNIQUE_OPERATORS", "LOC_TOTAL"
        ]
        
    def load_model(self, model_path: str = "model/model.pt"):
        """Load trained model and preprocessing parameters"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration
            feature_columns = checkpoint.get('feature_columns', self.standard_features)
            model_config = {
                'input_dim': len(feature_columns),
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
            self.feature_columns = feature_columns
            self.scaler_params = checkpoint['scaler_params']
            
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Input features ({len(self.feature_columns)}): {self.feature_columns}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess input data to match training format"""
        
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Ensure all expected features are present
        for col in self.feature_columns:
            if col not in df.columns:
                logger.warning(f"Missing feature {col}, setting to 0")
                df[col] = 0.0
        
        # Select and order features according to training
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
            input_data: Dictionary with software metrics values
            
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
        Generate realistic background data for software metrics SHAP explanations
        
        Args:
            num_samples: Number of background samples to generate
            
        Returns:
            Background data array
        """
        
        # Define realistic ranges for software metrics based on empirical studies
        feature_ranges = {
            'LOC_BLANK': (0, 50),
            'BRANCH_COUNT': (1, 30),
            'LOC_CODE_AND_COMMENT': (0, 10),
            'LOC_COMMENTS': (0, 50),
            'CYCLOMATIC_COMPLEXITY': (1, 25),
            'DESIGN_COMPLEXITY': (1, 20),
            'ESSENTIAL_COMPLEXITY': (1, 15),
            'LOC_EXECUTABLE': (5, 200),
            'HALSTEAD_CONTENT': (10, 100),
            'HALSTEAD_DIFFICULTY': (2, 40),
            'HALSTEAD_EFFORT': (100, 100000),
            'HALSTEAD_ERROR_EST': (0.01, 1.5),
            'HALSTEAD_LENGTH': (20, 500),
            'HALSTEAD_LEVEL': (0.01, 0.5),
            'HALSTEAD_PROG_TIME': (10, 5000),
            'HALSTEAD_VOLUME': (50, 3000),
            'NUM_OPERANDS': (5, 250),
            'NUM_OPERATORS': (10, 300),
            'NUM_UNIQUE_OPERANDS': (3, 80),
            'NUM_UNIQUE_OPERATORS': (5, 35),
            'LOC_TOTAL': (5, 250)
        }
        
        background_data = []
        
        for _ in range(num_samples):
            sample = []
            for feature in self.feature_columns:
                if feature in feature_ranges:
                    min_val, max_val = feature_ranges[feature]
                    
                    # Use log-normal distribution for some metrics that are typically skewed
                    if feature in ['HALSTEAD_EFFORT', 'HALSTEAD_VOLUME', 'LOC_EXECUTABLE']:
                        # Log-normal distribution for heavy-tailed metrics
                        value = np.random.lognormal(np.log(min_val + 1), 0.8)
                        value = np.clip(value, min_val, max_val)
                    elif 'COMPLEXITY' in feature:
                        # Exponential distribution for complexity metrics
                        value = np.random.exponential(3) + min_val
                        value = np.clip(value, min_val, max_val)
                    else:
                        # Uniform distribution for other metrics
                        value = np.random.uniform(min_val, max_val)
                else:
                    # Default range for unknown features
                    value = np.random.uniform(0, 10)
                
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
        
        for i, input_data in enumerate(input_list):
            try:
                result = self.predict(input_data)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(input_list)} predictions")
                    
            except Exception as e:
                logger.error(f"Error predicting for input {i}: {str(e)}")
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
            
            # Calculate SHAP values for test data (use subset for speed)
            test_subset = test_data[:min(50, len(test_data))]
            shap_values = explainer.shap_values(test_subset, nsamples=30)
            
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
            logger.info(f"Top 5 features: {list(importance_dict.keys())[:5]}")
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            # Return uniform importance as fallback
            uniform_importance = 1.0 / len(self.feature_columns)
            return {feature: uniform_importance for feature in self.feature_columns}
    
    def interpret_prediction(self, prediction_result: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide human-readable interpretation of prediction results
        
        Args:
            prediction_result: Result from predict()
            input_data: Original input data
            
        Returns:
            Dictionary with interpretation
        """
        
        interpretation = {
            'risk_level': 'Unknown',
            'confidence_level': 'Unknown',
            'key_risk_factors': [],
            'recommendations': [],
            'summary': ''
        }
        
        # Determine risk level
        reliability_score = prediction_result['reliability_score']
        if reliability_score < 0.3:
            interpretation['risk_level'] = 'High Risk'
        elif reliability_score < 0.7:
            interpretation['risk_level'] = 'Medium Risk'
        else:
            interpretation['risk_level'] = 'Low Risk'
        
        # Determine confidence level
        confidence = prediction_result['model_confidence']
        if confidence > 0.8:
            interpretation['confidence_level'] = 'High Confidence'
        elif confidence > 0.6:
            interpretation['confidence_level'] = 'Medium Confidence'
        else:
            interpretation['confidence_level'] = 'Low Confidence'
        
        # Identify key risk factors from SHAP values
        shap_values = prediction_result['shap_values']
        positive_contributors = {k: v for k, v in shap_values.items() if v > 0}
        top_risk_factors = sorted(positive_contributors.items(), 
                                key=lambda x: abs(x[1]), reverse=True)[:3]
        
        interpretation['key_risk_factors'] = [
            f"{factor}: {impact:.3f}" for factor, impact in top_risk_factors
        ]
        
        # Generate recommendations
        if interpretation['risk_level'] == 'High Risk':
            interpretation['recommendations'] = [
                "Consider code review and refactoring",
                "Increase testing coverage for this module",
                "Break down complex functions if possible",
                "Add more documentation and comments"
            ]
        elif interpretation['risk_level'] == 'Medium Risk':
            interpretation['recommendations'] = [
                "Monitor this module during testing",
                "Consider additional unit tests",
                "Review complex logic paths"
            ]
        else:
            interpretation['recommendations'] = [
                "Module appears low risk",
                "Maintain current quality practices"
            ]
        
        # Create summary
        fault_label = prediction_result['fault_label']
        fault_text = "likely to contain defects" if fault_label == 1 else "unlikely to contain defects"
        
        interpretation['summary'] = (
            f"This software module is {fault_text} "
            f"({interpretation['risk_level']}) with {interpretation['confidence_level']}. "
            f"Reliability score: {reliability_score:.2f}"
        )
        
        return interpretation