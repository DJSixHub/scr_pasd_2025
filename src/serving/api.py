"""
Model serving API using Ray Serve.
"""

import os
import ray
from ray import serve
import numpy as np
import pandas as pd
import time
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from flask import Flask, request, jsonify
import json

from src.utils.config import get_project_root

# Flask application for serving predictions
app = Flask(__name__)


@serve.deployment(route_prefix="/api")
class ModelDeployment:
    """
    Ray Serve deployment for serving machine learning models.
    """
    
    def __init__(self, config):
        """
        Initialize the model deployment.
        
        Args:
            config (Dict): Configuration for model deployment.
        """
        self.config = config
        self.models_dir = get_project_root() / 'models'
        self.loaded_models = {}
        self._load_models()
    
    def _load_models(self):
        """
        Load all available models from the models directory.
        """
        # Get paths of all .joblib files
        model_paths = list(self.models_dir.glob('*.joblib'))
        
        if not model_paths:
            print("Warning: No models found in models directory.")
            return
        
        # Load each model
        for model_path in model_paths:
            model_id = model_path.stem
            try:
                model = joblib.load(model_path)
                self.loaded_models[model_id] = model
                print(f"Loaded model: {model_id}")
            except Exception as e:
                print(f"Error loading model {model_id}: {e}")
    
    def list_models(self):
        """
        List all available models.
        
        Returns:
            List[str]: List of available model IDs.
        """
        return list(self.loaded_models.keys())
    
    def predict(self, model_id, features):
        """
        Make predictions using a specified model.
        
        Args:
            model_id (str): ID of the model to use.
            features (Dict): Feature values for prediction.
            
        Returns:
            Dict: Prediction results.
        """
        if model_id not in self.loaded_models:
            return {"error": f"Model {model_id} not found"}
        
        model = self.loaded_models[model_id]
        
        try:
            # Convert features to DataFrame or array
            if isinstance(features, dict):
                X = pd.DataFrame([features])
            else:
                X = np.array(features)
            
            # Make prediction
            start_time = time.time()
            y_pred = model.predict(X)
            latency = time.time() - start_time
            
            # Try to get prediction probabilities if available
            try:
                probabilities = model.predict_proba(X).tolist()
            except:
                probabilities = None
            
            # Return results
            result = {
                "prediction": y_pred.tolist(),
                "latency_ms": latency * 1000
            }
            
            if probabilities:
                result["probabilities"] = probabilities
                
            return result
        
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    async def __call__(self, request):
        """
        Handle incoming requests to the deployment.
        
        Args:
            request: The request object.
            
        Returns:
            Dict: Response data.
        """
        route = request.url.path.strip("/api")
        
        if route == "" or route == "/":
            return {"available_endpoints": ["/models", "/predict/<model_id>"]}
        
        elif route == "/models":
            return {"models": self.list_models()}
        
        elif route.startswith("/predict/"):
            model_id = route.split("/predict/")[1]
            try:
                data = await request.json()
                if not data or "features" not in data:
                    return {"error": "No features provided"}
                    
                return self.predict(model_id, data["features"])
            except Exception as e:
                return {"error": f"Invalid request: {str(e)}"}
        
        return {"error": "Invalid endpoint"}


class ModelServing:
    """
    Main class for model serving.
    """
    
    def __init__(self, config):
        """
        Initialize the model serving.
        
        Args:
            config (Dict): Configuration for model serving.
        """
        self.config = config
    
    def start(self):
        """
        Start the model serving.
        """
        print("Starting model serving...")
        
        # Initialize Ray Serve
        serve.start(detached=True)
        
        # Deploy the model service
        ModelDeployment.deploy(self.config)
        
        print(f"Model serving API started. The API is accessible at: http://localhost:8000/api")
        print("Available endpoints:")
        print("  - GET /api/models - List all available models")
        print("  - POST /api/predict/<model_id> - Make predictions with a model")

        # Keep the service running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down model serving...")
            serve.shutdown()
            print("Model serving stopped.")
