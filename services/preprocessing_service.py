import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import structlog

from config.settings import settings

logger = structlog.get_logger()

class PreprocessingService:
    def __init__(self):
        self.upload_dir = Path(settings.upload_directory)
    
    def preprocess_dataset(
        self,
        file_path: str,
        missing_strategy: str,
        scaling: bool,
        encoding: str,
        target_column: Optional[str] = None,
        selected_features: Optional[List[str]] = None
    ) -> str:
        """Preprocess a dataset with specified parameters"""
        try:
            # Load dataset
            df = pd.read_csv(file_path)
            original_filename = Path(file_path).name
            
            # Apply feature selection
            if selected_features:
                available_features = [col for col in selected_features if col in df.columns]
                if target_column and target_column not in available_features:
                    available_features.append(target_column)
                df = df[available_features]
            
            # Handle missing values
            df = self._handle_missing_values(df, missing_strategy)
            
            # Separate features and target
            if target_column and target_column in df.columns:
                X = df.drop(columns=[target_column])
                y = df[target_column]
            else:
                X = df
                y = None
            
            # Preprocessing pipeline
            X_processed = self._apply_preprocessing(X, scaling, encoding, y, target_column)
            
            # Reconstruct dataframe
            if y is not None:
                df_processed = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)
            else:
                df_processed = X_processed
            
            # Save preprocessed data
            preprocessed_filename = f"preprocessed_{original_filename}"
            preprocessed_path = self.upload_dir / preprocessed_filename
            df_processed.to_csv(preprocessed_path, index=False)
            
            logger.info(f"Preprocessing completed: {preprocessed_filename}")
            return str(preprocessed_path)
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise Exception(f"Preprocessing failed: {str(e)}")
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values in dataframe"""
        df_copy = df.copy()
        
        if strategy == "mean":
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
        elif strategy == "median":
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
        elif strategy == "mode":
            for col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode().iloc[0] if len(df_copy[col].mode()) > 0 else df_copy[col])
        elif strategy == "drop":
            df_copy = df_copy.dropna()
        
        return df_copy
    
    def _apply_preprocessing(
        self, 
        X: pd.DataFrame, 
        scaling: bool, 
        encoding: str, 
        y: Optional[pd.Series] = None,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """Apply scaling and encoding to features"""
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Build preprocessing pipeline
        transformers = []
        
        # Numeric preprocessing
        if numeric_cols:
            if scaling:
                transformers.append(('num', StandardScaler(), numeric_cols))
            else:
                transformers.append(('num', 'passthrough', numeric_cols))
        
        # Categorical preprocessing
        if categorical_cols:
            if encoding == "onehot":
                transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols))
            elif encoding == "label":
                # Apply label encoding manually
                X_copy = X.copy()
                for col in categorical_cols:
                    le = LabelEncoder()
                    X_copy[col] = le.fit_transform(X_copy[col].astype(str))
                transformers.append(('cat', 'passthrough', categorical_cols))
                X = X_copy
            else:  # fallback to onehot
                transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols))
        
        if not transformers:
            return X
        
        # Apply transformations
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        X_transformed = preprocessor.fit_transform(X)
        
        # Get feature names
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat' and encoding == "onehot":
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(cols))
                else:
                    feature_names.extend(cols)
            else:
                feature_names.extend(cols)
        
        return pd.DataFrame(X_transformed, columns=feature_names)
    
    def suggest_missing_strategy(self, df: pd.DataFrame) -> str:
        """Suggest best missing value strategy for dataset"""
        missing_counts = df.isnull().sum()
        total_rows = len(df)
        
        if missing_counts.sum() == 0:
            return 'mean'
        
        missing_percentages = missing_counts / total_rows
        
        # If more than 50% missing in any column, suggest drop
        if any(missing_percentages > 0.5):
            return 'drop'
        
        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # For numeric columns, check skewness
        if any(numeric_cols.isin(missing_counts[missing_counts > 0].index)):
            numeric_with_missing = numeric_cols.intersection(missing_counts[missing_counts > 0].index)
            skewness = df[numeric_with_missing].skew().abs()
            if any(skewness > 1):
                return 'median'
            return 'mean'
        
        # For categorical columns
        if any(categorical_cols.isin(missing_counts[missing_counts > 0].index)):
            return 'mode'
        
        return 'mean'
