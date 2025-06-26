"""
Model training utilities for distributed machine learning with fault tolerance
"""
import os
import ray
import logging
import numpy as np
import pandas as pd
import tempfile
from time import time
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import joblib
import random
import hashlib
import socket
from functools import partial

logger = logging.getLogger(__name__)

@ray.remote(max_restarts=3, max_task_retries=3)  # Add fault tolerance with retries
def train_model(model, X_train, y_train, X_test, y_test, model_name=None, checkpoint_dir=None):
    """
    Train a single model in a distributed manner using Ray with checkpointing
    
    Args:
        model: Scikit-learn model to train
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for logging
        checkpoint_dir: Directory to save checkpoints for fault tolerance
        
    Returns:
        dict: Training results including trained model and metrics
    """
    if model_name is None:
        model_name = model.__class__.__name__
    
    # Create a unique ID for this training job for checkpointing
    job_id = f"{model_name}_{socket.gethostname()}_{random.randint(1000, 9999)}"
    
    logger.info(f"Training model: {model_name} (job_id: {job_id})")
    
    checkpoint_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{job_id}.pkl")
    
    start_time = time()
    
    try:
        # Train the model
        model.fit(X_train, y_train)
        training_time = time() - start_time
        
        # Save checkpoint if enabled
        if checkpoint_path:
            try:
                joblib.dump(model, checkpoint_path)
                logger.info(f"Model checkpoint saved to {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
        
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

def train_multiple_models(models, X_train, y_train, X_test, y_test, model_names=None, checkpoint_dir=None, max_retries=3, timeout=600):
    """
    Train multiple models in parallel using Ray with fault tolerance
    
    Args:
        models (list): List of scikit-learn models to train
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_names (list, optional): Names of the models
        checkpoint_dir (str, optional): Directory to save checkpoints
        max_retries (int): Maximum number of retries for failed training
        timeout (int): Timeout in seconds for each training task
        
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
    
    # Create temporary checkpoint directory if not provided
    temp_checkpoint_dir = None
    if checkpoint_dir is None:
        temp_checkpoint_dir = tempfile.mkdtemp(prefix="ray_model_checkpoints_")
        checkpoint_dir = temp_checkpoint_dir
        logger.info(f"Created temporary checkpoint directory: {checkpoint_dir}")
    
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
                    model_name=model_name, checkpoint_dir=checkpoint_dir
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
    
    # Clean up temporary directory if created
    if temp_checkpoint_dir:
        try:
            import shutil
            shutil.rmtree(temp_checkpoint_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary checkpoint directory: {e}")
    
    if not trained_models:
        logger.error("All model training failed")
    else:
        logger.info(f"Successfully trained {len(trained_models)} models")
        
    return trained_models, accuracy_dict

@ray.remote
def save_model_to_path(model, model_name, output_path):
    """
    Save a single model to a specific path in a distributed manner.
    
    Args:
        model: The trained model to save
        model_name: Name of the model
        output_path: Full path where the model should be saved
        
    Returns:
        str or None: Path to the saved model or None if saving failed
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(model, output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error saving model {model_name} to {output_path}: {e}")
        return None

def save_models(models, output_dir="models", replicas=2):
    """
    Save trained models to disk with redundancy
    
    Args:
        models (dict): Dictionary mapping model names to trained models
        output_dir (str): Primary directory to save models in
        replicas (int): Number of copies to save for redundancy
        
    Returns:
        dict: Dictionary mapping model names to lists of saved file paths
    """
    if not models:
        logger.warning("No models to save")
        return {}
        
    if not ray.is_initialized():
        logger.warning("Ray is not initialized, initializing now")
        ray.init()
    
    # Create multiple directories for redundancy
    storage_paths = [output_dir]
    
    # Add replica directories if requested
    if replicas > 1:
        for i in range(1, replicas):
            replica_dir = f"{output_dir}_replica_{i}"
            storage_paths.append(replica_dir)
    
    # Ensure all directories exist
    for path in storage_paths:
        os.makedirs(path, exist_ok=True)
        
    # Dictionary to track save operations
    save_tasks = {}
    saved_paths = {}
    
    # Distribute saving across Ray workers
    for model_name, model in models.items():
        save_tasks[model_name] = []
        saved_paths[model_name] = []
        
        # Generate model hash for integrity verification
        model_hash = None
        try:
            import pickle
            model_bytes = pickle.dumps(model)
            model_hash = hashlib.md5(model_bytes).hexdigest()
        except:
            pass
            
        # Save to each storage location
        for path in storage_paths:
            file_path = os.path.join(path, f"{model_name}.joblib")
            
            # Add metadata for verification
            if model_hash:
                metadata_path = os.path.join(path, f"{model_name}.meta")
                with open(metadata_path, 'w') as f:
                    f.write(f"model_hash: {model_hash}\n")
                    f.write(f"timestamp: {time()}\n")
            
            # Submit save task to Ray
            save_task = save_model_to_path.remote(model, model_name, file_path)
            save_tasks[model_name].append(save_task)
    
    # Wait for all save operations to complete
    for model_name, tasks in save_tasks.items():
        try:
            paths = ray.get(tasks)
            saved_paths[model_name] = [p for p in paths if p]
            
            if saved_paths[model_name]:
                logger.info(f"Saved model {model_name} to {len(saved_paths[model_name])} locations")
            else:
                logger.error(f"Failed to save model {model_name} to any location")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
    
    return saved_paths

@ray.remote
def load_model_from_path(path):
    """
    Load a single model from a specific path in a distributed manner.
    
    Args:
        path: Path to the model file
        
    Returns:
        tuple: (model_name, model) or (model_name, None) if loading failed
    """
    try:
        model_name = os.path.basename(path).split('.')[0]
        model = joblib.load(path)
        return model_name, model
    except Exception as e:
        logger.error(f"Error loading model from {path}: {e}")
        return os.path.basename(path).split('.')[0], None

def verify_model_integrity(model_path):
    """
    Verify the integrity of a saved model using its metadata file
    
    Args:
        model_path: Path to the model file
        
    Returns:
        bool: True if the model passes integrity check
    """
    try:
        metadata_path = model_path.replace('.joblib', '.meta')
        if not os.path.exists(metadata_path):
            return True  # No metadata to verify against
            
        # Read stored hash
        stored_hash = None
        with open(metadata_path, 'r') as f:
            for line in f:
                if line.startswith('model_hash:'):
                    stored_hash = line.split(':')[1].strip()
                    break
                    
        if not stored_hash:
            return True  # No hash to verify against
            
        # Calculate current hash
        import pickle
        model = joblib.load(model_path)
        model_bytes = pickle.dumps(model)
        current_hash = hashlib.md5(model_bytes).hexdigest()
        
        return current_hash == stored_hash
    except Exception as e:
        logger.error(f"Error verifying model integrity for {model_path}: {e}")
        return False

def load_models(model_paths, verify_integrity=True):
    """
    Load models from disk in a fault-tolerant way
    
    Args:
        model_paths (dict or list): Dictionary mapping model names to lists of file paths,
                                   or a simple list of paths
        verify_integrity (bool): Whether to verify model integrity
        
    Returns:
        dict: Dictionary mapping model names to loaded models
    """
    if not model_paths:
        logger.warning("No model paths provided")
        return {}
        
    if not ray.is_initialized():
        logger.warning("Ray is not initialized, initializing now")
        ray.init()
    
    # Convert list to dictionary format if needed
    if isinstance(model_paths, list):
        paths_dict = {}
        for path in model_paths:
            model_name = os.path.basename(path).split('.')[0]
            if model_name not in paths_dict:
                paths_dict[model_name] = []
            paths_dict[model_name].append(path)
        model_paths = paths_dict
    
    # Try to load models with fault tolerance
    loaded_models = {}
    load_tasks = {}
    
    # For each model, try loading from multiple locations if available
    for model_name, paths in model_paths.items():
        load_tasks[model_name] = []
        
        # Check integrity first if requested
        valid_paths = []
        for path in paths:
            if os.path.exists(path):
                if not verify_integrity or verify_model_integrity(path):
                    valid_paths.append(path)
                else:
                    logger.warning(f"Model {path} failed integrity check, skipping")
            else:
                logger.warning(f"Model path {path} doesn't exist, skipping")
                
        # Try loading each valid copy
        for path in valid_paths:
            load_tasks[model_name].append(load_model_from_path.remote(path))
                
        if not load_tasks[model_name]:
            logger.error(f"No valid paths found for model {model_name}")
    
    # Collect results
    for model_name, tasks in load_tasks.items():
        for task in tasks:
            try:
                name, model = ray.get(task, timeout=30)
                if model is not None:
                    loaded_models[model_name] = model
                    logger.info(f"Successfully loaded model {model_name}")
                    break  # Stop after first successful load
            except Exception as e:
                logger.warning(f"Failed to load a copy of {model_name}: {e}")
    
    logger.info(f"Loaded {len(loaded_models)} models successfully")
    return loaded_models
