"""
Model serving functionality for distributed ML platform using FastAPI (Ray Serve deployment handled externally)
"""
import logging
import time
import pandas as pd
from typing import Dict, List, Any
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse

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

# Model predictor class (no Ray Serve logic here)
class ModelPredictor:
    def __init__(self, models):
        self.models = models
        self.request_times = {}
        self.request_counts = {}
        logger.info(f"ModelPredictor initialized with {len(models)} models")
    
    def _record_request(self, model_name, latency):
        if model_name not in self.request_times:
            self.request_times[model_name] = []
            self.request_counts[model_name] = 0
        self.request_times[model_name].append(latency)
        self.request_counts[model_name] += 1
        if len(self.request_times[model_name]) > 100:
            self.request_times[model_name] = self.request_times[model_name][-100:]
    
    def health(self, _=None):
        return {
            'status': 'healthy',
            'models_loaded': len(self.models),
            'model_names': list(self.models.keys())
        }
    
    def list_models(self, _=None):
        return {
            'models': list(self.models.keys())
        }
    
    async def predict(self, request_dict, model_name: str):
        start_time = time.time()
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail={
                'error': f'Model {model_name} not found',
                'available_models': list(self.models.keys())
            })
        try:
            try:
                prediction_request = PredictionFeatures(**request_dict)
            except Exception as e:
                raise HTTPException(status_code=400, detail={'error': f'Invalid request format: {str(e)}'})
            features = pd.DataFrame(prediction_request.features)
            result = self.models[model_name].predict(features)
            latency = time.time() - start_time
            self._record_request(model_name, latency)
            return {
                'model': model_name,
                'predictions': result.tolist(),
                'latency_ms': latency * 1000
            }
        except Exception as e:
            logger.error(f"Error making prediction with model {model_name}: {e}")
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail={'error': str(e)})
    
    def metrics(self, _=None):
        return {
            'request_counts': self.request_counts,
            'average_latency_ms': {
                model: (sum(times) / len(times) * 1000 if times else 0) 
                for model, times in self.request_times.items()
            }
        }
    
    def add_model(self, model_name, model):
        self.models[model_name] = model
        logger.info(f"Added model {model_name} to the server")
    
    def remove_model(self, model_name):
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Removed model {model_name} from the server")
            return True
        return False

def create_app(models):
    """
    Create a FastAPI app and ModelPredictor instance for Ray Serve deployment.
    Args:
        models (dict): Dictionary mapping model names to trained models
    Returns:
        app (FastAPI): FastAPI app instance
        predictor (ModelPredictor): ModelPredictor instance
    """
    predictor = ModelPredictor(models)
    app = FastAPI()

    @app.get("/health")
    async def health():
        return predictor.health()

    @app.get("/models")
    async def list_models():
        return predictor.list_models()

    @app.post("/predict/{model_name}")
    async def predict(request: Request, model_name: str):
        request_dict = await request.json()
        return await predictor.predict(request_dict, model_name)

    @app.get("/metrics")
    async def metrics():
        return predictor.metrics()

    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)}
        )

    return app, predictor