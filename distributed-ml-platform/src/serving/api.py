"""
Model serving functionality for distributed ML platform using FastAPI (Ray Serve deployment handled externally)
"""
import logging
import time
from typing import Dict, List, Any
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse
import ray

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
    def __init__(self, model_names):
        self.model_names = model_names
        self.request_times = {}
        self.request_counts = {}
        logger.info(f"ModelPredictor initialized with {len(model_names)} distributed models")
    
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
            'models_loaded': len(self.model_names),
            'model_names': list(self.model_names)
        }
    
    def list_models(self, _=None):
        return {
            'models': list(self.model_names)
        }
    
    async def predict(self, request_dict, model_name: str):
        import pandas as pd
        import time
        start_time = time.time()
        if model_name not in self.model_names:
            raise HTTPException(status_code=404, detail={
                'error': f'Model {model_name} not found',
                'available_models': list(self.model_names)
            })
        try:
            try:
                prediction_request = PredictionFeatures(**request_dict)
            except Exception as e:
                raise HTTPException(status_code=400, detail={'error': f'Invalid request format: {str(e)}'})
            features = prediction_request.features
            # Get the Ray actor by name and call predict
            try:
                actor = ray.get_actor(model_name)
            except Exception:
                raise HTTPException(status_code=404, detail={'error': f'Model actor {model_name} not found in Ray cluster'})
            result = ray.get(actor.predict.remote(features))
            latency = time.time() - start_time
            self._record_request(model_name, latency)
            return {
                'model': model_name,
                'predictions': result,
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

def create_app(model_names):
    """
    Create a FastAPI app and ModelPredictor instance for Ray Serve deployment.
    Args:
        models (dict): Dictionary mapping model names to trained models
    Returns:
        app (FastAPI): FastAPI app instance
        predictor (ModelPredictor): ModelPredictor instance
    """
    predictor = ModelPredictor(model_names)
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

    @app.get("/model-metrics/{model_name}")
    async def get_model_metrics(model_name: str):
        """Get training metrics for a specific model"""
        if model_name not in predictor.model_names:
            raise HTTPException(status_code=404, detail={
                'error': f'Model {model_name} not found',
                'available_models': list(predictor.model_names)
            })
        
        try:
            actor = ray.get_actor(model_name)
            metrics = ray.get(actor.get_metrics.remote())
            return {
                'model': model_name,
                'metrics': metrics
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail={'error': str(e)})

    @app.get("/all-metrics")
    async def get_all_metrics():
        """Get training metrics for all models"""
        all_metrics = {}
        for model_name in predictor.model_names:
            try:
                actor = ray.get_actor(model_name)
                metrics = ray.get(actor.get_metrics.remote())
                all_metrics[model_name] = metrics
            except Exception as e:
                all_metrics[model_name] = {'error': str(e)}
        
        return {'all_metrics': all_metrics}

    @app.get("/model-plots/{model_name}")
    async def get_model_plots(model_name: str):
        """Get ROC curve and learning curve data for a specific model"""
        if model_name not in predictor.model_names:
            raise HTTPException(status_code=404, detail={
                'error': f'Model {model_name} not found',
                'available_models': list(predictor.model_names)
            })
        
        try:
            actor = ray.get_actor(model_name)
            plot_data = ray.get(actor.generate_plot_data.remote())
            return plot_data
        except Exception as e:
            raise HTTPException(status_code=500, detail={'error': str(e)})

    @app.get("/all-plots")
    async def get_all_plots():
        """Get ROC curve and learning curve data for all models"""
        all_plots = {}
        for model_name in predictor.model_names:
            try:
                actor = ray.get_actor(model_name)
                plot_data = ray.get(actor.generate_plot_data.remote())
                all_plots[model_name] = plot_data
            except Exception as e:
                all_plots[model_name] = {'error': str(e)}
        
        return {'all_plots': all_plots}

    @app.get("/model-plots-png/{model_name}")
    async def get_model_plots_png(model_name: str):
        """Get ROC curve and learning curve as PNG images for a specific model"""
        if model_name not in predictor.model_names:
            raise HTTPException(status_code=404, detail={
                'error': f'Model {model_name} not found',
                'available_models': list(predictor.model_names)
            })
        
        try:
            actor = ray.get_actor(model_name)
            plot_pngs = ray.get(actor.generate_plots_png.remote())
            return plot_pngs
        except Exception as e:
            raise HTTPException(status_code=500, detail={'error': str(e)})

    @app.get("/model-roc-png/{model_name}")
    async def get_model_roc_png(model_name: str):
        """Get ROC curve as PNG image for a specific model"""
        if model_name not in predictor.model_names:
            raise HTTPException(status_code=404, detail={
                'error': f'Model {model_name} not found',
                'available_models': list(predictor.model_names)
            })
        
        try:
            actor = ray.get_actor(model_name)
            roc_png = ray.get(actor.generate_roc_png.remote())
            return roc_png
        except Exception as e:
            raise HTTPException(status_code=500, detail={'error': str(e)})

    @app.get("/model-learning-curve-png/{model_name}")
    async def get_model_learning_curve_png(model_name: str):
        """Get learning curve as PNG image for a specific model"""
        if model_name not in predictor.model_names:
            raise HTTPException(status_code=404, detail={
                'error': f'Model {model_name} not found',
                'available_models': list(predictor.model_names)
            })
        
        try:
            actor = ray.get_actor(model_name)
            learning_png = ray.get(actor.generate_learning_curve_png.remote())
            return learning_png
        except Exception as e:
            raise HTTPException(status_code=500, detail={'error': str(e)})

    @app.get("/all-plots-png")
    async def get_all_plots_png():
        """Get ROC curves and learning curves as PNG images for all models"""
        all_plots_png = {}
        for model_name in predictor.model_names:
            try:
                actor = ray.get_actor(model_name)
                plot_pngs = ray.get(actor.generate_plots_png.remote())
                all_plots_png[model_name] = plot_pngs
            except Exception as e:
                all_plots_png[model_name] = {'error': str(e)}
        
        return {'all_plots_png': all_plots_png}

    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)}
        )

    return app, predictor