"""
NeuroTrust Data Processor
Handles preprocessing of uploaded CSV datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data preprocessing and feature engineering for NeuroTrust"""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.label_encoders = {}
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
        logger.info(f"Columns: {list(df.columns)}")
        
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Clean column names
        df_processed.columns = df_processed.columns.str.strip().str.lower()
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Feature engineering
        df_processed = self.engineer_features(df_processed)
        
        # Identify target column
        target_column = self.identify_target_column(df_processed)
        
        # Separate features and target
        X, y = self.separate_features_target(df_processed, target_column)
        
        # Encode categorical features
        X_encoded = self.encode_categorical_features(X)
        
        # Scale numerical features
        X_scaled, scaler_params = self.scale_features(X_encoded)
        
        # Store feature information
        self.feature_columns = list(X_scaled.columns)
        
        logger.info(f"Processed dataset: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
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
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features from existing ones"""
        
        logger.info("Engineering features...")
        
        # Create complexity-related features
        if 'lines_of_code' in df.columns and 'file_complexity' in df.columns:
            df['complexity_per_line'] = df['file_complexity'] / (df['lines_of_code'] + 1)
        
        # Create commit-related features
        if 'recent_commits' in df.columns:
            df['commit_frequency'] = np.log1p(df['recent_commits'])
        
        # Create defect density if both lines_of_code and defects exist
        if 'lines_of_code' in df.columns and 'defects' in df.columns:
            df['defect_density'] = df['defects'] / (df['lines_of_code'] + 1)
        
        # Create experience score
        if 'last_editor_experience' in df.columns:
            experience_map = {
                'junior': 0,
                'mid': 1, 
                'senior': 2,
                0: 0,  # Handle numeric values
                1: 1,
                2: 2
            }
            df['experience_score'] = df['last_editor_experience'].map(experience_map).fillna(0)
        
        # Create risk score (if we have enough features)
        risk_features = ['file_complexity', 'defects', 'lines_of_code']
        available_risk_features = [col for col in risk_features if col in df.columns]
        
        if len(available_risk_features) >= 2:
            # Normalize features and create weighted risk score
            risk_scores = []
            weights = {'file_complexity': 0.4, 'defects': 0.4, 'lines_of_code': 0.2}
            
            for idx, row in df.iterrows():
                risk_score = 0
                total_weight = 0
                
                for feature in available_risk_features:
                    if feature in weights:
                        weight = weights[feature]
                        value = row[feature]
                        
                        # Normalize value (simple min-max within the column)
                        col_min = df[feature].min()
                        col_max = df[feature].max()
                        if col_max > col_min:
                            normalized_value = (value - col_min) / (col_max - col_min)
                        else:
                            normalized_value = 0
                        
                        risk_score += weight * normalized_value
                        total_weight += weight
                
                if total_weight > 0:
                    risk_scores.append(risk_score / total_weight)
                else:
                    risk_scores.append(0)
            
            df['risk_score'] = risk_scores
        
        logger.info(f"Feature engineering completed. New columns: {df.columns.tolist()}")
        
        return df
    
    def identify_target_column(self, df: pd.DataFrame) -> str:
        """Identify the target column for prediction"""
        
        # Common target column names
        potential_targets = [
            'defects', 'fault', 'bug', 'error', 'failure', 
            'fault_label', 'is_fault', 'has_defect', 'target', 'label'
        ]
        
        # Look for exact matches first
        for col in potential_targets:
            if col in df.columns:
                logger.info(f"Found target column: {col}")
                return col
        
        # Look for partial matches
        for col in df.columns:
            for target in potential_targets:
                if target in col.lower():
                    logger.info(f"Found target column (partial match): {col}")
                    return col
        
        # If no target found, check for binary columns
        binary_cols = []
        for col in df.columns:
            unique_vals = df[col].nunique()
            if unique_vals == 2:
                binary_cols.append(col)
        
        if binary_cols:
            target_col = binary_cols[0]  # Take the first binary column
            logger.info(f"Using first binary column as target: {target_col}")
            return target_col
        
        # Default: create target from defects if it exists
        if 'defects' in df.columns:
            df['fault_label'] = (df['defects'] > 0).astype(int)
            logger.info("Created binary target 'fault_label' from 'defects' column")
            return 'fault_label'
        
        # Last resort: use the last column
        target_col = df.columns[-1]
        logger.warning(f"No clear target found. Using last column as target: {target_col}")
        return target_col
    
    def separate_features_target(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Separate features and target variable"""
        
        # Create target array
        y = df[target_column].values
        
        # Convert target to binary if needed
        unique_values = np.unique(y)
        if len(unique_values) > 2:
            # Convert to binary (>0 means fault)
            if np.issubdtype(y.dtype, np.number):
                y = (y > 0).astype(int)
            else:
                # For non-numeric, encode as 0/1
                le = LabelEncoder()
                y = le.fit_transform(y)
                # If more than 2 classes, binarize
                if len(np.unique(y)) > 2:
                    y = (y > 0).astype(int)
        
        # Ensure binary values are 0 and 1
        y = y.astype(int)
        
        # Remove target column and prepare features
        feature_columns = [col for col in df.columns if col != target_column]
        X = df[feature_columns].copy()
        
        # Remove non-informative columns
        X = self.remove_non_informative_columns(X)
        
        logger.info(f"Features: {list(X.columns)}")
        logger.info(f"Target: {target_column} with {len(np.unique(y))} unique values")
        
        return X, y
    
    def remove_non_informative_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns that don't provide useful information"""
        
        columns_to_remove = []
        
        for col in df.columns:
            # Remove columns with only one unique value
            if df[col].nunique() <= 1:
                columns_to_remove.append(col)
                continue
            
            # Remove columns that are mostly null
            if df[col].isnull().sum() / len(df) > 0.9:
                columns_to_remove.append(col)
                continue
            
            # Remove text columns that look like IDs or names
            if df[col].dtype == 'object':
                if any(keyword in col.lower() for keyword in ['id', 'name', 'module', 'file', 'timestamp']):
                    # Keep if it looks like a useful categorical variable
                    if df[col].nunique() > len(df) * 0.8:  # Too many unique values
                        columns_to_remove.append(col)
        
        if columns_to_remove:
            logger.info(f"Removing non-informative columns: {columns_to_remove}")
            df = df.drop(columns=columns_to_remove)
        
        return df
    
    def encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        
        X_encoded = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                logger.info(f"Encoding categorical column: {col}")
                
                # Use label encoding for ordinal-like categories
                if col in ['last_editor_experience', 'experience']:
                    experience_map = {'junior': 0, 'mid': 1, 'senior': 2}
                    X_encoded[col] = X[col].map(experience_map).fillna(0)
                else:
                    # Use label encoding for other categorical variables
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
        
        return X_encoded
    
    def scale_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Scale numerical features"""
        
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
        
        logger.info("Feature scaling completed")
        
        return X_scaled_df, scaler_params
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean input data for prediction"""
        
        validated_data = input_data.copy()
        
        # Ensure all required features are present
        required_features = ['file_complexity', 'recent_commits', 'defects', 'lines_of_code']
        
        for feature in required_features:
            if feature not in validated_data:
                logger.warning(f"Missing required feature: {feature}, setting to 0")
                validated_data[feature] = 0
        
        # Validate data types and ranges
        if 'file_complexity' in validated_data:
            validated_data['file_complexity'] = max(0, float(validated_data['file_complexity']))
        
        if 'recent_commits' in validated_data:
            validated_data['recent_commits'] = max(0, int(validated_data['recent_commits']))
        
        if 'defects' in validated_data:
            validated_data['defects'] = max(0, int(validated_data['defects']))
        
        if 'lines_of_code' in validated_data:
            validated_data['lines_of_code'] = max(1, int(validated_data['lines_of_code']))
        
        if 'last_editor_experience' in validated_data:
            valid_experiences = ['junior', 'mid', 'senior']
            if validated_data['last_editor_experience'] not in valid_experiences:
                logger.warning(f"Invalid experience level, defaulting to 'mid'")
                validated_data['last_editor_experience'] = 'mid'
        
        return validated_data