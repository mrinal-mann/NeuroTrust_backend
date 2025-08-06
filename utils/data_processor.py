"""
NeuroTrust Data Processor - UPDATED FOR REAL DATASETS
Handles preprocessing of NASA MDP software metrics datasets (CM1, JM1, etc.)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data preprocessing for NeuroTrust with real software metrics datasets"""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.label_encoders = {}
        
        # Standard feature set for software defect prediction (21 common features)
        self.standard_features = [
            "LOC_BLANK", "BRANCH_COUNT", "LOC_CODE_AND_COMMENT", "LOC_COMMENTS",
            "CYCLOMATIC_COMPLEXITY", "DESIGN_COMPLEXITY", "ESSENTIAL_COMPLEXITY", 
            "LOC_EXECUTABLE", "HALSTEAD_CONTENT", "HALSTEAD_DIFFICULTY", 
            "HALSTEAD_EFFORT", "HALSTEAD_ERROR_EST", "HALSTEAD_LENGTH", 
            "HALSTEAD_LEVEL", "HALSTEAD_PROG_TIME", "HALSTEAD_VOLUME",
            "NUM_OPERANDS", "NUM_OPERATORS", "NUM_UNIQUE_OPERANDS", 
            "NUM_UNIQUE_OPERATORS", "LOC_TOTAL"
        ]
        
        self.feature_columns = []
    
    def process_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process uploaded dataset for training
        
        Args:
            df: Raw pandas DataFrame from CSV
            
        Returns:
            Dictionary containing processed features, targets, and metadata
        """
        
        logger.info(f"Processing dataset with shape {df.shape}")
        logger.info(f"Original columns: {list(df.columns)}")
        
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Clean column names - remove whitespace but preserve case for software metrics
        df_processed.columns = df_processed.columns.str.strip()
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Identify target column
        target_column = self.identify_target_column(df_processed)
        logger.info(f"Identified target column: {target_column}")
        
        # Separate features and target
        X, y = self.separate_features_target(df_processed, target_column)
        
        # Ensure we have the standard feature set
        X_standardized = self.standardize_features(X)
        
        # Scale numerical features
        X_scaled, scaler_params = self.scale_features(X_standardized)
        
        # Store feature information
        self.feature_columns = list(X_scaled.columns)
        
        logger.info(f"Processed dataset: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        logger.info(f"Final features: {self.feature_columns}")
        logger.info(f"Target distribution: {np.bincount(y.astype(int))}")
        
        return {
            'features': X_scaled.values,
            'targets': y,
            'feature_columns': self.feature_columns,
            'scaler_params': scaler_params,
            'target_column': target_column,
            'dataset_info': {
                'n_samples': len(df_processed),
                'n_features': len(self.feature_columns),
                'target_distribution': np.bincount(y.astype(int)).tolist()
            }
        }
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        
        logger.info("Handling missing values...")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Separate numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            # Impute numeric columns with median
            if len(numeric_cols) > 0:
                numeric_imputer = SimpleImputer(strategy='median')
                df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
            
            # Impute categorical columns with mode
            if len(categorical_cols) > 0:
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
        else:
            logger.info("No missing values detected")
        
        return df
    
    def identify_target_column(self, df: pd.DataFrame) -> str:
        """Identify the target column for defect prediction"""
        
        # Common target column names in NASA MDP and similar datasets
        potential_targets = [
            'defect_label', 'defects', 'fault', 'bug', 'error', 'failure',
            'fault_label', 'is_fault', 'has_defect', 'target', 'label',
            'Defective', 'defective'  # CM1 style
        ]
        
        # Look for exact matches first (case-sensitive)
        for col in df.columns:
            if col in potential_targets:
                logger.info(f"Found target column (exact match): {col}")
                return col
        
        # Look for case-insensitive matches
        for col in df.columns:
            for target in potential_targets:
                if col.lower() == target.lower():
                    logger.info(f"Found target column (case-insensitive): {col}")
                    return col
        
        # Look for partial matches
        for col in df.columns:
            col_lower = col.lower()
            if any(target in col_lower for target in ['defect', 'fault', 'bug', 'label']):
                logger.info(f"Found target column (partial match): {col}")
                return col
        
        # Check for binary columns (likely targets)
        binary_cols = []
        for col in df.columns:
            unique_vals = df[col].nunique()
            if unique_vals == 2:
                unique_values = df[col].unique()
                # Check if it looks like a target (Y/N, 0/1, True/False, etc.)
                if set(str(v).upper() for v in unique_values).issubset({'Y', 'N', '1', '0', 'TRUE', 'FALSE'}):
                    binary_cols.append(col)
        
        if binary_cols:
            target_col = binary_cols[0]
            logger.info(f"Using binary column as target: {target_col}")
            return target_col
        
        # Last resort: use the last column
        target_col = df.columns[-1]
        logger.warning(f"No clear target found. Using last column: {target_col}")
        return target_col
    
    def separate_features_target(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Separate features and target variable"""
        
        # Create target array
        y = df[target_column].values
        
        # Convert target to binary 0/1 format
        if y.dtype == 'object' or isinstance(y[0], str):
            # Handle string targets (Y/N, True/False, etc.)
            y_str = [str(val).upper().strip() for val in y]
            y = np.array([1 if val in ['Y', 'YES', 'TRUE', '1', 'DEFECTIVE'] else 0 for val in y_str])
        else:
            # Handle numeric targets
            unique_values = np.unique(y)
            if len(unique_values) > 2:
                # Multi-class: convert to binary (>0 means defect)
                y = (y > 0).astype(int)
            else:
                # Already binary, ensure 0/1 format
                y = y.astype(int)
        
        logger.info(f"Target conversion completed. Unique values: {np.unique(y)}")
        logger.info(f"Target distribution: Class 0: {sum(y == 0)}, Class 1: {sum(y == 1)}")
        
        # Remove target column and prepare features
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns].copy()
        
        logger.info(f"Original features: {len(X.columns)}")
        logger.info(f"Available features: {list(X.columns)}")
        
        return X, y
    
    def standardize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Standardize feature set to the 21 common software metrics"""
        
        logger.info("Standardizing features to common set...")
        
        X_std = pd.DataFrame()
        
        # Ensure all standard features are present
        for feature in self.standard_features:
            if feature in X.columns:
                X_std[feature] = X[feature]
                logger.info(f"✓ Found feature: {feature}")
            else:
                # Feature missing, set to 0 (or could use median imputation)
                X_std[feature] = 0.0
                logger.warning(f"✗ Missing feature: {feature}, setting to 0")
        
        # Add any additional features that might be useful but not in standard set
        additional_features = []
        for col in X.columns:
            if col not in self.standard_features:
                # Only add if it's numeric and has reasonable variance
                if X[col].dtype in ['int64', 'float64'] and X[col].nunique() > 1:
                    additional_features.append(col)
        
        if additional_features:
            logger.info(f"Found {len(additional_features)} additional features: {additional_features}")
            # For now, we'll stick to the standard 21 features for consistency
            # But this could be extended based on needs
        
        logger.info(f"Standardized to {len(X_std.columns)} features")
        return X_std
    
    def scale_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Scale numerical features using StandardScaler"""
        
        logger.info("Scaling features...")
        
        # Initialize scaler
        scaler = StandardScaler()
        
        # Fit and transform
        X_scaled = scaler.fit_transform(X)
        
        # Create DataFrame with original column names
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Store scaler parameters
        scaler_params = {
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist(),
            'feature_names': X.columns.tolist()
        }
        
        # Log feature scaling info
        logger.info("Feature scaling completed")
        logger.info(f"Feature ranges after scaling: min={X_scaled_df.min().min():.3f}, max={X_scaled_df.max().max():.3f}")
        
        return X_scaled_df, scaler_params
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean input data for prediction"""
        
        validated_data = input_data.copy()
        
        # Ensure all standard features are present
        for feature in self.standard_features:
            if feature not in validated_data:
                logger.warning(f"Missing feature: {feature}, setting to 0")
                validated_data[feature] = 0.0
        
        # Validate data types and reasonable ranges for software metrics
        feature_validators = {
            'LOC_BLANK': lambda x: max(0, float(x)),
            'BRANCH_COUNT': lambda x: max(0, int(float(x))),
            'LOC_CODE_AND_COMMENT': lambda x: max(0, float(x)),
            'LOC_COMMENTS': lambda x: max(0, float(x)),
            'CYCLOMATIC_COMPLEXITY': lambda x: max(1, float(x)),
            'DESIGN_COMPLEXITY': lambda x: max(1, float(x)),
            'ESSENTIAL_COMPLEXITY': lambda x: max(1, float(x)),
            'LOC_EXECUTABLE': lambda x: max(1, float(x)),
            'HALSTEAD_CONTENT': lambda x: max(0, float(x)),
            'HALSTEAD_DIFFICULTY': lambda x: max(0.1, float(x)),
            'HALSTEAD_EFFORT': lambda x: max(0, float(x)),
            'HALSTEAD_ERROR_EST': lambda x: max(0, float(x)),
            'HALSTEAD_LENGTH': lambda x: max(1, float(x)),
            'HALSTEAD_LEVEL': lambda x: max(0.001, min(1.0, float(x))),
            'HALSTEAD_PROG_TIME': lambda x: max(0, float(x)),
            'HALSTEAD_VOLUME': lambda x: max(0, float(x)),
            'NUM_OPERANDS': lambda x: max(0, int(float(x))),
            'NUM_OPERATORS': lambda x: max(0, int(float(x))),
            'NUM_UNIQUE_OPERANDS': lambda x: max(1, int(float(x))),
            'NUM_UNIQUE_OPERATORS': lambda x: max(1, int(float(x))),
            'LOC_TOTAL': lambda x: max(1, float(x))
        }
        
        # Apply validation
        for feature, validator in feature_validators.items():
            if feature in validated_data:
                try:
                    validated_data[feature] = validator(validated_data[feature])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {feature}: {validated_data[feature]}, setting to default")
                    # Set reasonable defaults based on feature type
                    if 'LOC' in feature or feature in ['BRANCH_COUNT', 'HALSTEAD_LENGTH']:
                        validated_data[feature] = 10.0
                    elif 'COMPLEXITY' in feature:
                        validated_data[feature] = 2.0
                    elif 'HALSTEAD' in feature:
                        validated_data[feature] = 1.0
                    else:
                        validated_data[feature] = 1.0
        
        return validated_data
      
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """Get domain knowledge weights for software defect prediction features"""
        
        # Based on software engineering research and domain expertise
        importance_weights = {
            'CYCLOMATIC_COMPLEXITY': 0.15,  # Very important
            'LOC_EXECUTABLE': 0.12,          # High correlation with defects
            'HALSTEAD_DIFFICULTY': 0.10,     # Important Halstead metric
            'HALSTEAD_EFFORT': 0.08,         # Effort indicates complexity
            'BRANCH_COUNT': 0.08,            # Control flow complexity
            'DESIGN_COMPLEXITY': 0.07,       # Design quality indicator
            'ESSENTIAL_COMPLEXITY': 0.07,    # Structural complexity
            'HALSTEAD_VOLUME': 0.06,         # Code volume indicator
            'LOC_TOTAL': 0.06,              # Size metric
            'NUM_OPERATORS': 0.05,           # Operator complexity
            'NUM_OPERANDS': 0.05,            # Operand complexity
            'HALSTEAD_ERROR_EST': 0.04,      # Direct error estimate
            'NUM_UNIQUE_OPERATORS': 0.03,    # Vocabulary diversity
            'NUM_UNIQUE_OPERANDS': 0.03,     # Vocabulary diversity
            'LOC_COMMENTS': 0.02,            # Documentation quality
            'HALSTEAD_LENGTH': 0.02,         # Program length
            'HALSTEAD_CONTENT': 0.02,        # Information content
            'LOC_CODE_AND_COMMENT': 0.01,    # Mixed content lines
            'HALSTEAD_LEVEL': 0.01,          # Abstraction level
            'LOC_BLANK': 0.01,               # Formatting indicator
            'HALSTEAD_PROG_TIME': 0.01       # Development time estimate
        }
        
        # Normalize to sum to 1
        total_weight = sum(importance_weights.values())
        normalized_weights = {k: v / total_weight for k, v in importance_weights.items()}
        
        return normalized_weights
    
    def validate_dataset_compatibility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate if dataset is compatible with NeuroTrust"""
        
        compatibility_report = {
            'is_compatible': True,
            'issues': [],
            'recommendations': [],
            'feature_coverage': 0.0,
            'detected_format': 'unknown'
        }
        
        # Check for common NASA MDP dataset formats
        columns_lower = [col.lower() for col in df.columns]
        
        if any('defective' in col.lower() for col in df.columns):
            compatibility_report['detected_format'] = 'CM1/KC1 format'
        elif 'label' in columns_lower:
            compatibility_report['detected_format'] = 'JM1 format'
        elif any('defect' in col.lower() for col in df.columns):
            compatibility_report['detected_format'] = 'Custom defect format'
        
        # Check feature coverage
        available_features = [f for f in self.standard_features if f in df.columns]
        coverage = len(available_features) / len(self.standard_features)
        compatibility_report['feature_coverage'] = coverage
        
        if coverage < 0.5:
            compatibility_report['is_compatible'] = False
            compatibility_report['issues'].append(
                f"Low feature coverage: {coverage:.1%} ({len(available_features)}/{len(self.standard_features)})"
            )
        
        # Check for target variable
        target_found = False
        for col in df.columns:
            if any(target in col.lower() for target in ['defect', 'fault', 'label', 'bug']):
                target_found = True
                break
        
        if not target_found:
            compatibility_report['issues'].append("No clear target variable found")
        
        # Check sample size
        if len(df) < 50:
            compatibility_report['issues'].append(f"Dataset too small: {len(df)} samples (minimum 50 recommended)")
        
        # Generate recommendations
        if coverage < 0.8:
            missing_features = [f for f in self.standard_features if f not in df.columns]
            compatibility_report['recommendations'].append(
                f"Missing important features: {missing_features[:5]}..."
            )
        
        if len(df) < 200:
            compatibility_report['recommendations'].append(
                "Consider combining with additional datasets for better model performance"
            )
        
        return compatibility_report