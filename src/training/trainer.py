###### Implementación de entrenamiento distribuido usando Ray

import os
import ray
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import joblib
from pathlib import Path

from src.utils.data_loader import load_dataset
from src.utils.config import get_project_root
from src.models.model_factory import get_model


@ray.remote
class TrainingWorker:
    ###### Actor de Ray para entrenar modelos en paralelo
    
    def __init__(self, model_config):
        ###### Inicializar el trabajador de entrenamiento
        #
        # Args:
        #    model_config (Dict): Configuración para el entrenamiento del modelo
        self.model_config = model_config
        self.model_type = model_config.get('type', 'random_forest')
        self.model_params = model_config.get('params', {})
        self.model = None
    
    def train(self, X_train, y_train, model_id):
        """
        Train a model on the given data.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            model_id (str): Identifier for the model.
            
        Returns:
            Dict: Training results.
        """
        # Create the model
        self.model = get_model(self.model_type, **self.model_params)
        
        # Record training time
        start_time = time.time()
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        return {
            'model_id': model_id,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'training_time': training_time,
        }
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dict: Evaluation results.
        """
        if self.model is None:
            return {'error': 'Model not trained'}
        
        # Predict on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics based on problem type
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
        
        try:
            # Try classification metrics first
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            }
        except:
            # Fall back to regression metrics
            metrics = {
                'r2': float(r2_score(y_test, y_pred)),
                'mse': float(mean_squared_error(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred)))
            }
        
        return metrics
    
    def save_model(self, save_path):
        """
        Save the trained model to disk.
        
        Args:
            save_path (str): Path to save the model.
            
        Returns:
            bool: True if successful.
        """
        if self.model is None:
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, save_path)
        return True
    
    def get_model(self):
        """
        Get the trained model.
        
        Returns:
            BaseEstimator: The trained model.
        """
        return self.model


class DistributedTrainer:
    """
    Main class for distributed model training using Ray.
    """
    
    def __init__(self, config):
        """
        Initialize the distributed trainer.
        
        Args:
            config (Dict): Configuration for the trainer.
        """
        self.config = config
        self.datasets = config.get('datasets', [])
        self.models = config.get('models', [])
        self.results = {}
        self.models_dir = get_project_root() / 'models'
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
    
    def train(self):
        """
        Train models on all datasets in parallel.
        """
        print("Starting distributed training...")
        
        for dataset_config in self.datasets:
            dataset_name = dataset_config['name']
            target_column = dataset_config['target_column']
            
            print(f"Processing dataset: {dataset_name}")
            
            # Load and preprocess data
            train_df, test_df = load_dataset(dataset_name, 
                                           test_size=dataset_config.get('test_size', 0.2),
                                           random_state=dataset_config.get('random_state', 42),
                                           target_column=target_column)
            
            # Separate features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            # Train models in parallel
            workers = []
            for i, model_config in enumerate(self.models):
                model_id = f"{dataset_name}_{model_config['type']}_{i}"
                worker = TrainingWorker.remote(model_config)
                workers.append((worker, model_id))
            
            # Start training in parallel
            train_refs = []
            for worker, model_id in workers:
                train_ref = worker.train.remote(X_train, y_train, model_id)
                train_refs.append((worker, model_id, train_ref))
            
            # Wait for all training to complete
            dataset_results = []
            for worker, model_id, train_ref in train_refs:
                train_result = ray.get(train_ref)
                
                # Evaluate model
                eval_ref = worker.evaluate.remote(X_test, y_test)
                eval_result = ray.get(eval_ref)
                
                # Save model
                save_path = str(self.models_dir / f"{model_id}.joblib")
                save_ref = worker.save_model.remote(save_path)
                ray.get(save_ref)
                
                # Combine results
                result = {**train_result, **eval_result, 'save_path': save_path}
                dataset_results.append(result)
                
                print(f"Model {model_id} trained and evaluated:")
                print(f"  Training time: {result['training_time']:.2f} seconds")
                for metric, value in eval_result.items():
                    print(f"  {metric}: {value:.4f}")
            
            # Store results for this dataset
            self.results[dataset_name] = dataset_results
            
            print(f"Completed training for dataset: {dataset_name}")
        
        print("Distributed training completed!")
        return self.results
