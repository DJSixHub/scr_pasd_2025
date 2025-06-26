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
    
    def health(self, _):
        """Health check endpoint"""
        return {
            'status': 'healthy',
            'models_loaded': len(self.models),
            'model_names': list(self.models.keys())
        }
    
    def list_models(self, _):
        """List available models"""
        return {
            'models': list(self.models.keys())
        }
    
    async def predict(self, request_dict, model_name: str):
        """Make predictions using the specified model"""
        start_time = time.time()
        
        if model_name not in self.models:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail={
                'error': f'Model {model_name} not found',
                'available_models': list(self.models.keys())
            })
            
        try:
            # Validate request data
            try:
                prediction_request = PredictionFeatures(**request_dict)
            except Exception as e:
                from fastapi import HTTPException
                raise HTTPException(status_code=400, detail={'error': f'Invalid request format: {str(e)}'})
                
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
            from fastapi import HTTPException
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail={'error': str(e)})
    
    def metrics(self, _):
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
            
            # For Ray 2.7.1, the start API has changed
            serve.start(detached=True, host=self.host, port=self.port)
            
            # Deploy the model predictor
            predictor = ModelPredictor(self.models)
            
            # Updated deployment pattern for Ray Serve 2.7.1 using FastAPI for routing
            from fastapi import FastAPI, Request, HTTPException
            from starlette.responses import JSONResponse
            
            app = FastAPI()
            
            @app.get("/health")
            async def health():
                return predictor.health(None)
                
            @app.get("/models")
            async def list_models():
                return predictor.list_models(None)
                
            @app.get("/predict/{model_name}")
            async def predict(request: Request, model_name: str):
                request_dict = await request.json()
                return await predictor.predict(request_dict, model_name)
                
            @app.get("/metrics")
            async def metrics():
                return predictor.metrics(None)
            
            @app.exception_handler(Exception)
            async def exception_handler(request: Request, exc: Exception):
                return JSONResponse(
                    status_code=500,
                    content={"error": str(exc)}
                )
                
            # Deploy the FastAPI app with Ray Serve
            @serve.deployment(name="api", route_prefix="/")
            class APIDeployment:
                def __init__(self):
                    self.app = app
                    self.predictor = predictor
                    
                @property
                def predictor_ref(self):
                    return predictor
                
                async def __call__(self, request):
                    return await self.app(request)
            
            # Deploy the API
            api_deployment = APIDeployment.bind()
            serve.run(api_deployment)
            
            self.predictor_handle = predictor
            
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