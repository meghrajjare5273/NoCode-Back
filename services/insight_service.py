import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger()

class InsightService:
    """Service for generating dataset insights and suggestions"""
    
    VALID_TASK_TYPES = ["classification", "regression", "clustering"]
    
    def generate_insights(self, summary: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Generate insights and suggestions for a dataset"""
        try:
            insights = []
            suggested_task_type = "clustering"
            suggested_target_column = None
            
            # Basic dataset info
            insights.append(f"Dataset contains {summary['rows']} rows and {len(summary['columns'])} columns")
            
            # Missing values analysis
            missing_cols = [col for col, count in summary['missing_values'].items() if count > 0]
            if missing_cols:
                insights.append(f"Missing values detected in {len(missing_cols)} columns")
            else:
                insights.append("No missing values detected in the dataset")
            
            # Data types analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            insights.append(f"Dataset contains {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
            
            # Task type and target column suggestions
            suggested_task_type, suggested_target_column = self._suggest_ml_task(df, numeric_cols, categorical_cols)
            
            if suggested_target_column:
                unique_count = df[suggested_target_column].nunique()
                if suggested_task_type == "classification":
                    insights.append(f"'{suggested_target_column}' appears suitable for classification with {unique_count} classes")
                elif suggested_task_type == "regression":
                    insights.append(f"'{suggested_target_column}' appears suitable for regression (continuous values)")
            else:
                insights.append("No obvious target variable detected, clustering recommended for exploratory analysis")
            
            # Missing value strategy suggestion
            suggested_missing_strategy = self._suggest_missing_strategy(df, summary['missing_values'])
            
            return {
                "insights": insights,
                "suggested_task_type": suggested_task_type,
                "suggested_target_column": suggested_target_column,
                "suggested_missing_strategy": suggested_missing_strategy
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {
                "insights": ["Unable to generate detailed insights"],
                "suggested_task_type": "clustering",
                "suggested_target_column": None,
                "suggested_missing_strategy": "mean"
            }
    
    def _suggest_ml_task(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> tuple:
        """Suggest ML task type and target column"""
        
        # Check for classification targets (categorical columns with reasonable number of classes)
        for col in categorical_cols:
            unique_values = df[col].nunique()
            if 2 <= unique_values <= 10:  # Good for classification
                return "classification", col
        
        # Check for regression targets (numeric columns with many unique values)
        for col in numeric_cols:
            unique_values = df[col].nunique()
            if unique_values > 20:  # Good for regression
                # Check if values are reasonably distributed
                if df[col].std() > 0:  # Has variation
                    return "regression", col
        
        # Default to clustering if no clear target found
        return "clustering", None
    
    def _suggest_missing_strategy(self, df: pd.DataFrame, missing_values: Dict[str, int]) -> str:
        """Suggest best missing value handling strategy"""
        total_missing = sum(missing_values.values())
        if total_missing == 0:
            return "mean"
        
        total_rows = len(df)
        missing_percentage = total_missing / (total_rows * len(df.columns))
        
        # If too many missing values, suggest dropping
        if missing_percentage > 0.3:
            return "drop"
        
        # Check data types of columns with missing values
        numeric_cols_with_missing = []
        categorical_cols_with_missing = []
        
        for col, count in missing_values.items():
            if count > 0:
                if df[col].dtype in ['int64', 'float64']:
                    numeric_cols_with_missing.append(col)
                else:
                    categorical_cols_with_missing.append(col)
        
        # If mostly categorical columns have missing values
        if len(categorical_cols_with_missing) > len(numeric_cols_with_missing):
            return "mode"
        
        # For numeric columns, check skewness
        if numeric_cols_with_missing:
            try:
                skewness = df[numeric_cols_with_missing].skew().abs().mean()
                if skewness > 1:
                    return "median"
                else:
                    return "mean"
            except:
                return "mean"
        
        return "mean"
