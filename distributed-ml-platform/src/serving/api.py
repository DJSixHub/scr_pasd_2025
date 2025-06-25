"""
Model serving functionality for distributed ML platform using Ray Serve
"""
import ray
from ray import serve
import logging
import time
import pandas as pd
from typing import Dict, List, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Pydantic models for request validation
class PredictionFeatures(BaseModel):
    features: List[Dict[str, Any]]
    
class PredictionResponse(BaseModel):
    model: str
    predictions: List
    latency_ms: float
    
class ErrorResponse(BaseModel):
    error: str
    available_models: List[str] = None

# Ray Serve predictor class
class ModelPredictor:
    def __init__(self, models):
        self.models = models
        self.request_times = {}
        self.request_counts = {}
        logger.info(f"ModelPredictor initialized with {len(models)} models")
    
    def _record_request(self, model_name, latency):
        """
        Record request latency for metrics
        
        Args:
            model_name (str): Name of the model used
            latency (float): Request processing time in seconds
        """
        if model_name not in self.request_times:
            self.request_times[model_name] = []
            self.request_counts[model_name] = 0
            
        self.request_times[model_name].append(latency)
        self.request_counts[model_name] += 1
        
        # Keep only the last 100 request times to avoid memory issues
        if len(self.request_times[model_name]) > 100:
            self.request_times[model_name] = self.request_times[model_name][-100:]
    
    async def health(self, _):
        """Health check endpoint"""
        return {
            'status': 'healthy',
            'models_loaded': len(self.models),
            'model_names': list(self.models.keys())
        }
    
    async def list_models(self, _):
        """List available models"""
        return {
            'models': list(self.models.keys())
        }
    
    async def predict(self, request_dict, model_name: str):
        """Make predictions using the specified model"""
        start_time = time.time()
        
        if model_name not in self.models:
            return {
                'error': f'Model {model_name} not found',
                'available_models': list(self.models.keys())
            }, 404
            
        try:
            # Validate request data
            try:
                prediction_request = PredictionFeatures(**request_dict)
            except Exception as e:
                return {'error': f'Invalid request format: {str(e)}'}, 400
                
            features = pd.DataFrame(prediction_request.features)
            
            # Make prediction using the model
            result = self.models[model_name].predict(features)
            
            # Record latency
            latency = time.time() - start_time
            self._record_request(model_name, latency)
            
            return {
                'model': model_name,
                'predictions': result.tolist(),
                'latency_ms': latency * 1000
            }
            
        except Exception as e:
            logger.error(f"Error making prediction with model {model_name}: {e}")
            return {'error': str(e)}, 500
    
    async def metrics(self, _):
        """Get model serving metrics"""
        return {
            'request_counts': self.request_counts,
            'average_latency_ms': {
                model: (sum(times) / len(times) * 1000 if times else 0) 
                for model, times in self.request_times.items()
            }
        }
    
    def add_model(self, model_name, model):
        """
        Add a model to the server
        
        Args:
            model_name (str): Name of the model
            model: Trained model object
        """
        self.models[model_name] = model
        logger.info(f"Added model {model_name} to the server")
    
    def remove_model(self, model_name):
        """
        Remove a model from the server
        
        Args:
            model_name (str): Name of the model to remove
        """
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Removed model {model_name} from the server")
            return True
        return False

class ModelServer:
    def __init__(self, models=None, host="0.0.0.0", port=8000):
        """
        Initialize a model server using Ray Serve
        
        Args:
            models (dict): Dictionary mapping model names to trained models
            host (str): Host address to bind the server to
            port (int): Port to bind the server to
        """
        self.models = models or {}
        self.host = host
        self.port = port
        self.is_running = False
        self.predictor_handle = None
        
    def start(self, blocking=False):
        """
        Start the model server
        
        Args:
            blocking (bool): Whether to run in blocking mode
        """
        try:
            # Start Ray Serve
            if not ray.is_initialized():
                ray.init()
                
            serve.start(detached=True, http_options={"host": self.host, "port": self.port})
            
            # Deploy the model predictor
            predictor = ModelPredictor(self.models)
            predictor_deployment = serve.deployment(name="predictor")(lambda models: predictor)
            self.predictor_handle = predictor_deployment.deploy(self.models)
            
            # Set up routes
            serve.ingress("health", route_prefix="/health").bind(predictor.health)
            serve.ingress("models", route_prefix="/models").bind(predictor.list_models)
            serve.ingress("predict", route_prefix="/predict/{model_name}").bind(predictor.predict)
            serve.ingress("metrics", route_prefix="/metrics").bind(predictor.metrics)
            
            self.is_running = True
            logger.info(f"Started model server at http://{self.host}:{self.port}")
            
            return True
        except Exception as e:
            logger.error(f"Error starting Ray Serve: {e}")
            return False

    def stop(self):
        """
        Stop the model server if running
        """
        if self.is_running:
            try:
                serve.shutdown()
                self.is_running = False
                logger.info("Stopped model server")
                return True
            except Exception as e:
                logger.error(f"Error stopping Ray Serve: {e}")
                return False
        return True
        
    def add_model(self, model_name, model):
        """Add a model to the server"""
        if self.predictor_handle:
            ray.get(self.predictor_handle.add_model.remote(model_name, model))
            logger.info(f"Added model {model_name} to the server")
        else:
            logger.error("Cannot add model: server not running")
            
    def remove_model(self, model_name):
        """Remove a model from the server"""
        if self.predictor_handle:
            return ray.get(self.predictor_handle.remove_model.remote(model_name))
        else:
            logger.error("Cannot remove model: server not running")
            return False

def create_api(models=None, host="0.0.0.0", port=8000):
    """
    Create and start a model serving API using Ray Serve
    
    Args:
        models (dict): Dictionary mapping model names to trained models
        host (str): Host address to bind the server to
        port (int): Port to bind the server to
        
    Returns:
        ModelServer: The created server instance
    """
    server = ModelServer(models=models, host=host, port=port)
    server.start()
    return server