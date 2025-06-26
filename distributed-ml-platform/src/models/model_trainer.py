"""
Model training utilities for distributed machine learning with fault tolerance (Ray-native, no file I/O)
"""
import ray
import logging
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import random
import socket

logger = logging.getLogger(__name__)

@ray.remote(max_retries=3)
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
    
    # Create a unique ID for this training job
    job_id = f"{model_name}_{socket.gethostname()}_{random.randint(1000, 9999)}"
    
    logger.info(f"Training model: {model_name} (job_id: {job_id})")
    
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

def train_multiple_models(models, X_train, y_train, X_test, y_test, model_names=None, max_retries=3, timeout=600):
    """
    Train multiple models in parallel using Ray with fault tolerance
    
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
    
    logger.info(f"Training {len(models)} models in parallel with fault tolerance")
    
    # Data sharding - store data in Ray's object store for distributed access
    X_train_id = ray.put(X_train)
    y_train_id = ray.put(y_train)
    X_test_id = ray.put(X_test)
    y_test_id = ray.put(y_test)
    
    # Start distributed training with fault tolerance
    training_refs = []
    for i, (model, model_name) in enumerate(zip(models, model_names)):
        # Distribute models across different Ray workers
        model_id = ray.put(model)
        
        # Submit training task with retry logic
        for attempt in range(max_retries):
            try:
                ref = train_model.remote(
                    model_id, X_train_id, y_train_id, X_test_id, y_test_id, 
                    model_name=model_name
                )
                training_refs.append((ref, model_name, attempt))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to submit {model_name} training task (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                    time.sleep(1)
                else:
                    logger.error(f"Failed to submit {model_name} training task after {max_retries} attempts: {e}")
    
    # Collect results with timeout handling
    trained_models = {}
    accuracy_dict = {}
    
    for ref, model_name, _ in training_refs:
        try:
            # Wait for result with timeout
            result = ray.get(ref, timeout=timeout)
            
            if result.get('error'):
                logger.error(f"Model {model_name} failed: {result['error']}")
                continue
                
            trained_models[model_name] = result['model']
            
            # Extract metrics for comparison
            metrics = result['metrics']
            if 'accuracy' in metrics:
                accuracy_dict[model_name] = metrics['accuracy']
            elif 'rmse' in metrics:
                # For regression, use negative RMSE so that higher is better (consistent with accuracy)
                accuracy_dict[model_name] = -metrics['rmse']
        except ray.exceptions.GetTimeoutError:
            logger.error(f"Training {model_name} timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Error getting result for {model_name}: {e}")
    
    if not trained_models:
        logger.error("All model training failed")
    else:
        logger.info(f"Successfully trained {len(trained_models)} models")
        
    return trained_models, accuracy_dict

# Ray actor for distributed, in-memory model storage and prediction
@ray.remote(max_restarts=-1, max_task_retries=-1)
class ModelActor:
    def __init__(self, model, model_name):
        """
        Initialize the ModelActor with a scikit-learn model and its name.
        
        Args:
            model: The scikit-learn model to be used for predictions.
            model_name: A string representing the name of the model.
        """
        self.model = model
        self.model_name = model_name
        self.metrics = {}
    
    def predict(self, features):
        """
        Make predictions using the stored model.
        
        Args:
            features: Input features for prediction, as a list of dictionaries or a DataFrame.
            
        Returns:
            List of predictions.
        """
        import pandas as pd
        X = pd.DataFrame(features)
        preds = self.model.predict(X)
        return preds.tolist()
    
    def get_name(self):
        """
        Get the name of the model.
        
        Returns:
            The model name as a string.
        """
        return self.model_name
    
    def get_metrics(self):
        """
        Get the metrics of the model.
        
        Returns:
            A dictionary containing the model's metrics.
        """
        return self.metrics
    
    def set_metrics(self, metrics):
        """
        Set the metrics for the model.
        
        Args:
            metrics: A dictionary containing the metrics to be set for the model.
        """
        self.metrics = metrics
