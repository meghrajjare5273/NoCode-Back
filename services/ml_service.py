import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, calinski_harabasz_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import joblib
import structlog
import asyncio
import pandas as pd
import joblib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional
from pathlib import Path
import structlog

# from services.ml_service import MLService
from utils.exceptions import ModelTrainingError

from config.settings import settings

logger = structlog.get_logger()

class MLService:
    def __init__(self):
        self.upload_dir = Path(settings.upload_directory)
        self._setup_models()
    
    def _setup_models(self):
        """Initialize available models"""
        self.classification_models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=settings.random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=settings.random_state),
            'decision_tree': DecisionTreeClassifier(random_state=settings.random_state),
            'knn': KNeighborsClassifier(),
            'svm': SVC(probability=True, random_state=settings.random_state),
        }
        
        self.regression_models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=settings.random_state),
            'decision_tree': DecisionTreeRegressor(random_state=settings.random_state),
            'knn': KNeighborsRegressor(),
            'svm': SVR(),
        }
        
        self.clustering_models = {
            'kmeans': KMeans(n_clusters=3, random_state=settings.random_state, n_init=10),
            'dbscan': DBSCAN(),
        }
    
    def train_model(
        self,
        file_path: str,
        task_type: str,
        model_type: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train a machine learning model"""
        try:
            start_time = time.time()
            
            # Load preprocessed data
            df = pd.read_csv(file_path)
            
            # Select default model if not specified
            if not model_type:
                model_type = self._get_default_model(task_type)
            
            # Train based on task type
            if task_type == "classification":
                result = self._train_classification(df, model_type, target_column)
            elif task_type == "regression":
                result = self._train_regression(df, model_type, target_column)
            elif task_type == "clustering":
                result = self._train_clustering(df, model_type)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            # Add timing information
            result["training_time"] = time.time() - start_time
            
            # Save model
            if "model" in result:
                model_filename = f"trained_model_{Path(file_path).stem}.pkl"
                model_path = self.upload_dir / model_filename
                joblib.dump(result["model"], model_path)
                del result["model"]  # Remove model object from result
            
            logger.info(f"Model training completed: {task_type}/{model_type}")
            return result
            
        except Exception as e:
            logger.error(f"Model training error: {str(e)}")
            return {
                "task_type": task_type,
                "model_type": model_type,
                "results": {},
                "error": f"Training failed: {str(e)}"
            }
    
    def _train_classification(self, df: pd.DataFrame, model_type: str, target_column: str) -> Dict[str, Any]:
        """Train classification model"""
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        if model_type not in self.classification_models:
            raise ValueError(f"Unsupported classification model: {model_type}")
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.default_test_size, random_state=settings.random_state
        )
        
        # Train model
        model = self.classification_models[model_type]
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        }
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            results["cv_mean"] = float(cv_scores.mean())
            results["cv_std"] = float(cv_scores.std())
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, X.columns)
        
        return {
            "task_type": "classification",
            "model_type": model_type,
            "results": results,
            "feature_importance": feature_importance,
            "model": model
        }
    
    def _train_regression(self, df: pd.DataFrame, model_type: str, target_column: str) -> Dict[str, Any]:
        """Train regression model"""
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        if model_type not in self.regression_models:
            raise ValueError(f"Unsupported regression model: {model_type}")
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.default_test_size, random_state=settings.random_state
        )
        
        # Train model
        model = self.regression_models[model_type]
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results = {
            "r2_score": float(r2_score(y_test, y_pred)),
            "mean_squared_error": float(mean_squared_error(y_test, y_pred)),
            "mean_absolute_error": float(mean_absolute_error(y_test, y_pred))
        }
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            results["cv_mean"] = float(cv_scores.mean())
            results["cv_std"] = float(cv_scores.std())
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, X.columns)
        
        return {
            "task_type": "regression",
            "model_type": model_type,
            "results": results,
            "feature_importance": feature_importance,
            "model": model
        }
    
    def _train_clustering(self, df: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """Train clustering model"""
        if model_type not in self.clustering_models:
            raise ValueError(f"Unsupported clustering model: {model_type}")
        
        # Prepare data (use all columns for clustering)
        X = df.select_dtypes(include=[np.number])  # Only numeric columns
        if X.empty:
            raise ValueError("No numeric columns found for clustering")
        
        # Train model
        model = self.clustering_models[model_type]
        
        if hasattr(model, 'fit_predict'):
            labels = model.fit_predict(X)
        else:
            model.fit(X)
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
        
        # Calculate metrics
        results = {}
        
        if len(np.unique(labels)) > 1 and -1 not in labels:  # Valid clustering
            try:
                results["silhouette_score"] = float(silhouette_score(X, labels))
                results["calinski_harabasz_score"] = float(calinski_harabasz_score(X, labels))
            except Exception as e:
                logger.warning(f"Clustering metrics calculation failed: {str(e)}")
        
        # Model-specific metrics
        if model_type == "kmeans":
            results["inertia"] = float(model.inertia_)
            results["n_clusters"] = int(model.n_clusters)
        
        return {
            "task_type": "clustering",
            "model_type": model_type,
            "results": results,
            "model": model
        }
    
    def _get_default_model(self, task_type: str) -> str:
        """Get default model for task type"""
        defaults = {
            "classification": "logistic_regression",
            "regression": "linear_regression",
            "clustering": "kmeans"
        }
        return defaults.get(task_type, "logistic_regression")
    
    def _get_feature_importance(self, model, feature_names) -> Optional[List[List[Any]]]:
        """Extract feature importance from model"""
        try:
            importance_dict = {}
            
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) == 1:
                    importance_dict = dict(zip(feature_names, np.abs(model.coef_)))
                else:
                    importance_dict = dict(zip(feature_names, np.mean(np.abs(model.coef_), axis=0)))
            
            if importance_dict:
                # Sort by importance and return top 10
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                return sorted_importance[:10]
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return None
        




class AsyncMLService:
    def __init__(self):
        self.ml_service = MLService()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def train_model_async(
        self,
        file_path: str,
        task_type: str,
        model_type: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train model asynchronously in background thread"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.ml_service.train_model,
                file_path,
                task_type,
                model_type,
                target_column
            )
            return result
        except Exception as e:
            logger.error(f"Async training error: {str(e)}")
            raise ModelTrainingError(f"Async training failed: {str(e)}")
    
    async def predict_async(self, model_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._predict_sync,
                model_path,
                data
            )
            return result
        except Exception as e:
            logger.error(f"Async prediction error: {str(e)}")
            raise ModelTrainingError(f"Async prediction failed: {str(e)}")
    
    def _predict_sync(self, model_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous prediction method"""
        model = joblib.load(model_path)
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df).tolist()
        
        return {
            "prediction": prediction.tolist(),
            "probabilities": probabilities,
            "model_id": model_path.stem
        }

