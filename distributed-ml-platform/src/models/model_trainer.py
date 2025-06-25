"""
Model training utilities for distributed machine learning
"""
import os
import ray
import logging
import numpy as np
import pandas as pd
from time import time
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import joblib

logger = logging.getLogger(__name__)

@ray.remote
def train_model(model, X_train, y_train, X_test, y_test, model_name=None):
    """
    Train a single model in a distributed manner using Ray
    
    Args:
        model: Scikit-learn model to train
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for logging
        
    Returns:
        dict: Training results including trained model and metrics
    """
    if model_name is None:
        model_name = model.__class__.__name__
    
    logger.info(f"Training model: {model_name}")
    
    start_time = time()
    
    try:
        # Train the model
        model.fit(X_train, y_train)
        training_time = time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {}
        
        # Classification metrics
        if len(np.unique(y_train)) < 10:  # Assuming it's a classification task if fewer than 10 unique values
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        # Regression metrics
        else:
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
        metrics['training_time'] = training_time
        
        logger.info(f"Training completed for {model_name}: {metrics}")
        
        return {
            'model': model,
            'model_name': model_name,
            'metrics': metrics
        }
    except Exception as e:
        logger.error(f"Error training model {model_name}: {e}")
        return {
            'model': None,
            'model_name': model_name,
            'error': str(e)
        }

def train_multiple_models(models, X_train, y_train, X_test, y_test, model_names=None):
    """
    Train multiple models in parallel using Ray
    
    Args:
        models (list): List of scikit-learn models to train
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_names (list, optional): Names of the models
        
    Returns:
        tuple: (trained_models, accuracy_dict)
    """
    if not ray.is_initialized():
        logger.warning("Ray is not initialized, initializing now")
        ray.init()
    
    if model_names is None:
        model_names = [model.__class__.__name__ for model in models]
    
    logger.info(f"Training {len(models)} models in parallel")
    
    try:
        # Train models in parallel
        model_refs = [
            train_model.remote(model, X_train, y_train, X_test, y_test, name)
            for model, name in zip(models, model_names)
        ]
        
        # Get training results
        training_results = ray.get(model_refs)
        
        # Organize results
        trained_models = {}
        metrics = {}
        
        for result in training_results:
            model_name = result['model_name']
            
            if 'error' in result:
                logger.error(f"Model {model_name} training failed: {result['error']}")
                continue
                
            trained_models[model_name] = result['model']
            metrics[model_name] = result['metrics']
            
        logger.info(f"Successfully trained {len(trained_models)} models")
        
        return trained_models, metrics
    except Exception as e:
        logger.error(f"Error in train_multiple_models: {e}")
        return {}, {}

def save_models(models, output_dir="models"):
    """
    Save trained models to disk
    
    Args:
        models (dict): Dictionary mapping model names to trained models
        output_dir (str): Directory to save models in
        
    Returns:
        list: Paths to saved model files
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for model_name, model in models.items():
        try:
            file_path = os.path.join(output_dir, f"{model_name}.joblib")
            joblib.dump(model, file_path)
            saved_paths.append(file_path)
            logger.info(f"Saved model {model_name} to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
    
    return saved_paths

def load_models(model_paths):
    """
    Load models from disk
    
    Args:
        model_paths (list): Paths to model files
        
    Returns:
        dict: Dictionary mapping model names to loaded models
    """
    loaded_models = {}
    
    for path in model_paths:
        try:
            model_name = os.path.basename(path).split('.')[0]
            model = joblib.load(path)
            loaded_models[model_name] = model
            logger.info(f"Loaded model {model_name} from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
    
    return loaded_models
