"""
Factory for creating machine learning models.
"""

from typing import Dict, Any
import numpy as np
from sklearn.base import BaseEstimator


def get_model(model_type: str, **model_params) -> BaseEstimator:
    """
    Factory function to create a machine learning model.
    
    Args:
        model_type (str): Type of model to create.
        **model_params: Parameters for the model.
        
    Returns:
        BaseEstimator: A scikit-learn compatible model.
    """
    model_type = model_type.lower()
    
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        # Determine if it's classification or regression based on parameters
        if model_params.pop('task', 'classification') == 'classification':
            return RandomForestClassifier(**model_params)
        else:
            return RandomForestRegressor(**model_params)
    
    elif model_type == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        if model_params.pop('task', 'classification') == 'classification':
            return GradientBoostingClassifier(**model_params)
        else:
            return GradientBoostingRegressor(**model_params)
    
    elif model_type == 'xgboost':
        # Check if xgboost is installed
        try:
            import xgboost as xgb
            if model_params.pop('task', 'classification') == 'classification':
                return xgb.XGBClassifier(**model_params)
            else:
                return xgb.XGBRegressor(**model_params)
        except ImportError:
            print("Warning: XGBoost not installed. Falling back to Random Forest.")
            if model_params.pop('task', 'classification') == 'classification':
                return RandomForestClassifier(**model_params)
            else:
                return RandomForestRegressor(**model_params)
    
    elif model_type == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**model_params)
    
    elif model_type == 'linear_regression':
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**model_params)
    
    elif model_type == 'svm':
        from sklearn.svm import SVC, SVR
        if model_params.pop('task', 'classification') == 'classification':
            return SVC(**model_params)
        else:
            return SVR(**model_params)
    
    elif model_type == 'knn':
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        if model_params.pop('task', 'classification') == 'classification':
            return KNeighborsClassifier(**model_params)
        else:
            return KNeighborsRegressor(**model_params)
    
    elif model_type == 'decision_tree':
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        if model_params.pop('task', 'classification') == 'classification':
            return DecisionTreeClassifier(**model_params)
        else:
            return DecisionTreeRegressor(**model_params)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
