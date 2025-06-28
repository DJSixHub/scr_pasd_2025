"""
Simplified Model serving API for distributed ML platform using FastAPI
"""
import logging
import time
import base64
import io
import os
import subprocess
import asyncio
from typing import Dict, List, Any
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from starlette.responses import JSONResponse
import ray

logger = logging.getLogger(__name__)

# Pydantic models for request validation
class PredictionFeatures(BaseModel):
    features: List[Dict[str, Any]]

class DatasetPredictionFeatures(BaseModel):
    dataset: str  # Dataset name (e.g., "iris", "wine", "breast_cancer")
    features: List[Dict[str, Any]]
    
class PredictionResponse(BaseModel):
    model: str
    predictions: List
    latency_ms: float

class FileUploadRequest(BaseModel):
    filename: str
    content: str  # Base64 encoded file content
    
class FileUploadResponse(BaseModel):
    filename: str
    status: str
    rows: int = None
    columns: List[str] = None
    preview: List[Dict[str, Any]] = None
    
class ErrorResponse(BaseModel):
    error: str
    available_models: List[str] = None

# Model predictor class
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

    # ...existing code...
    
    async def predict_with_fallback(self, request_dict, model_name: str):
        """Predict with automatic fallback if actor is not available"""
        import pandas as pd
        import time
        start_time = time.time()
        
        try:
            prediction_request = PredictionFeatures(**request_dict)
        except Exception as e:
            raise HTTPException(status_code=400, detail={'error': f'Invalid request format: {str(e)}'})
        
        features = prediction_request.features
        
        # Try to get the actor with simple retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                actor = ray.get_actor(model_name)
                result = ray.get(actor.predict.remote(features), timeout=30.0)
                
                latency = time.time() - start_time
                self._record_request(model_name, latency)
                return {
                    'model': model_name,
                    'predictions': result,
                    'latency_ms': latency * 1000
                }
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {model_name}: {e}, retrying...")
                    time.sleep(1)
                else:
                    logger.error(f"Error making prediction with model {model_name}: {e}")
                    raise HTTPException(status_code=500, detail={'error': str(e)})

def create_app(model_names):
    """
    Create FastAPI app for model serving - Simplified API
    
    Args:
        model_names: List of available model names
        
    Returns:
        FastAPI app instance and predictor instance
    """
    predictor = ModelPredictor(model_names)
    app = FastAPI(title="Distributed ML Platform API", description="Simplified API for distributed machine learning")
    
    @app.get("/")
    async def root():
        return {"message": "Distributed ML Platform API", "models_available": len(model_names)}
    
    @app.get("/health")
    async def health():
        return predictor.health()

    # Development reload endpoint (distributed-friendly)
    @app.post("/reload")
    async def reload_api():
        """Reload the API server (for development) - distributed-friendly without volumes"""
        import os
        
        # Check if we're in development mode
        if os.getenv('DEVELOPMENT_MODE', '').lower() == 'true':
            # In a distributed system, we can signal the container to restart
            # This is safer than file watching in a distributed environment
            import signal
            import asyncio
            
            async def restart_server():
                await asyncio.sleep(1)  # Give time to return response
                os.kill(os.getpid(), signal.SIGTERM)
            
            # Schedule restart after returning response
            asyncio.create_task(restart_server())
            
            return {
                "message": "API server restart initiated",
                "note": "In distributed mode, container will restart to pick up changes"
            }
        else:
            return {
                "error": "Reload only available in development mode",
                "note": "Set DEVELOPMENT_MODE=true environment variable"
            }

    # 1. MODELS - List all models
    @app.get("/models")
    async def models():
        """List all available models"""
        return predictor.list_models()
    
    # 2. PREDICT(all) - Predict using all models
    @app.post("/predict/all")
    async def predict_all(request: PredictionFeatures):
        """Make predictions using all available models"""
        results = {}
        for model_name in model_names:
            try:
                result = await predictor.predict(request.dict(), model_name)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {'error': str(e)}
        return {'all_predictions': results}
    
    # 3. PREDICT(all, dataset) - Predict using all models trained on a specific dataset
    @app.post("/predict/all/{dataset}")
    async def predict_all_dataset(dataset: str, request: PredictionFeatures):
        """Make predictions using all models trained on a specific dataset"""
        results = {}
        dataset_models = [name for name in model_names if name.endswith(f"_{dataset}")]
        if not dataset_models:
            raise HTTPException(status_code=404, detail=f"No models found for dataset: {dataset}")
        
        for model_name in dataset_models:
            try:
                result = await predictor.predict(request.dict(), model_name)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {'error': str(e)}
        return {'dataset': dataset, 'predictions': results}

    # 4. PREDICT(model, dataset) - Predict using specific model and dataset with resilience
    @app.post("/predict/{model_type}/{dataset}")
    async def predict_model_dataset(model_type: str, dataset: str, request: PredictionFeatures):
        """Make predictions using a specific model type trained on a specific dataset with failover"""
        model_name = f"{model_type}_{dataset}"
        return await predictor.predict_with_fallback(request.dict(), model_name)
    
    # 5. PREDICT(model) - Predict using specific model with resilience
    @app.post("/predict/{model_name}")
    async def predict_model(model_name: str, request: PredictionFeatures):
        """Make predictions using a specific model with automatic failover"""
        return await predictor.predict_with_fallback(request.dict(), model_name)

    # 5. METRICS(model) - Get metrics for specific model
    @app.get("/metrics/{model_name}")
    async def metrics_model(model_name: str):
        """Get training metrics for a specific model"""
        try:
            actor = ray.get_actor(model_name)
            metrics = ray.get(actor.get_metrics.remote())
            return {'model_name': model_name, 'metrics': metrics}
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found: {str(e)}")

    # 6. METRICS(all) - Get metrics for all models
    @app.get("/metrics/all")
    async def metrics_all():
        """Get training metrics for all models"""
        all_metrics = {}
        for model_name in model_names:
            try:
                actor = ray.get_actor(model_name)
                metrics = ray.get(actor.get_metrics.remote())
                all_metrics[model_name] = metrics
            except Exception as e:
                all_metrics[model_name] = {'error': str(e)}
        return {'all_metrics': all_metrics}

    # 7. VISUALIZATION(model) - View PNG files for specific model as images
    @app.get("/visualization/{model_name}/roc")
    async def get_roc_curve(model_name: str):
        """Get ROC curve as a viewable PNG image"""
        try:
            actor = ray.get_actor(model_name)
            roc_png_data = ray.get(actor.generate_roc_png.remote())
            
            if 'error' in roc_png_data:
                raise HTTPException(status_code=500, detail=roc_png_data['error'])
            
            # Decode base64 to binary PNG data
            png_bytes = base64.b64decode(roc_png_data['roc_curve_png'])
            
            # Return as streaming response with proper headers
            return StreamingResponse(
                io.BytesIO(png_bytes),
                media_type="image/png",
                headers={"Content-Disposition": f"inline; filename={model_name}_roc_curve.png"}
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found: {str(e)}")

    @app.get("/visualization/{model_name}/learning_curve")
    async def get_learning_curve(model_name: str):
        """Get learning curve as a viewable PNG image"""
        try:
            actor = ray.get_actor(model_name)
            learning_png_data = ray.get(actor.generate_learning_curve_png.remote())
            
            if 'error' in learning_png_data:
                raise HTTPException(status_code=500, detail=learning_png_data['error'])
            
            # Decode base64 to binary PNG data
            png_bytes = base64.b64decode(learning_png_data['learning_curve_png'])
            
            # Return as streaming response with proper headers
            return StreamingResponse(
                io.BytesIO(png_bytes),
                media_type="image/png",
                headers={"Content-Disposition": f"inline; filename={model_name}_learning_curve.png"}
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found: {str(e)}")

    # 7. VISUALIZATIONS - Generate plots for models
    @app.get("/plot/{model_name}/roc")
    async def plot_roc_curve(model_name: str):
        """Generate ROC curve for a specific model"""
        try:
            actor = ray.get_actor(model_name)
            plot_data = ray.get(actor.generate_roc_curve.remote())
            return plot_data
        except Exception as e:
            raise HTTPException(status_code=404, detail={'error': f'Could not generate ROC curve for {model_name}: {str(e)}'})

    @app.get("/plot/{model_name}/learning")
    async def plot_learning_curve(model_name: str):
        """Generate learning curve for a specific model"""
        try:
            actor = ray.get_actor(model_name)
            plot_data = ray.get(actor.generate_learning_curve.remote())
            return plot_data
        except Exception as e:
            raise HTTPException(status_code=404, detail={'error': f'Could not generate learning curve for {model_name}: {str(e)}'})

    @app.get("/plot/{model_name}/roc/png")
    async def plot_roc_curve_png(model_name: str):
        """Generate ROC curve as PNG image for a specific model"""
        try:
            actor = ray.get_actor(model_name)
            png_data = ray.get(actor.generate_roc_png.remote())
            if 'error' in png_data:
                raise HTTPException(status_code=500, detail=png_data)
            
            # Return base64 PNG image
            return {
                'model_name': model_name,
                'image_base64': png_data['roc_curve_png'],
                'format': 'png'
            }
        except Exception as e:
            raise HTTPException(status_code=404, detail={'error': f'Could not generate ROC curve PNG for {model_name}: {str(e)}'})

    @app.get("/plot/{model_name}/learning/png")
    async def plot_learning_curve_png(model_name: str):
        """Generate learning curve as PNG image for a specific model"""
        try:
            actor = ray.get_actor(model_name)
            png_data = ray.get(actor.generate_learning_curve_png.remote())
            if 'error' in png_data:
                raise HTTPException(status_code=500, detail=png_data)
            
            # Return base64 PNG image
            return {
                'model_name': model_name,
                'image_base64': png_data['learning_curve_png'],
                'format': 'png'
            }
        except Exception as e:
            raise HTTPException(status_code=404, detail={'error': f'Could not generate learning curve PNG for {model_name}: {str(e)}'})

    @app.get("/plot/{model_name}/all")
    async def plot_all_data(model_name: str):
        """Generate all plot data for a specific model"""
        try:
            actor = ray.get_actor(model_name)
            plot_data = ray.get(actor.generate_plot_data.remote())
            return plot_data
        except Exception as e:
            raise HTTPException(status_code=404, detail={'error': f'Could not generate plot data for {model_name}: {str(e)}'})

    # 8. VISUALIZATION(all) - HTML dashboard showing all models  
    @app.get("/visualization/all")
    async def visualization_all():
        """Get HTML dashboard showing visualizations for all models"""
        
        # Group models by dataset
        datasets = {}
        for model_name in model_names:
            if '_' in model_name:
                dataset = model_name.split('_', 1)[1]
                if dataset not in datasets:
                    datasets[dataset] = []
                datasets[dataset].append(model_name)
        
        # Generate HTML content
        models_html = ""
        for dataset, models in datasets.items():
            models_html += f"""
            <div class="dataset-section">
                <h2>Dataset: {dataset}</h2>
                <div class="models-grid">
            """
            for model_name in models:
                model_type = model_name.split('_')[0]
                models_html += f"""
                    <div class="model-card">
                        <h3>{model_type}</h3>
                        <div class="chart-links">
                            <a href="/visualization/{model_name}" target="_blank">View Full Dashboard</a>
                        </div>
                        <div class="charts">
                            <div class="chart">
                                <h4>ROC Curve</h4>
                                <img src="/visualization/{model_name}/roc" alt="ROC Curve">
                            </div>
                            <div class="chart">
                                <h4>Learning Curve</h4>
                                <img src="/visualization/{model_name}/learning_curve" alt="Learning Curve">
                            </div>
                        </div>
                    </div>
                """
            models_html += """
                </div>
            </div>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Platform - All Model Visualizations</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f2f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; }}
                h1 {{ color: #333; text-align: center; border-bottom: 3px solid #4CAF50; padding-bottom: 20px; }}
                .dataset-section {{ background-color: white; margin: 30px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .dataset-section h2 {{ color: #2c5f41; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
                .models-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; margin-top: 20px; }}
                .model-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #fafafa; }}
                .model-card h3 {{ color: #555; text-align: center; margin-bottom: 15px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
                .chart-links {{ text-align: center; margin-bottom: 15px; }}
                .chart-links a {{ background-color: #4CAF50; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; font-size: 14px; }}
                .chart-links a:hover {{ background-color: #45a049; }}
                .charts {{ display: flex; gap: 10px; }}
                .chart {{ flex: 1; text-align: center; }}
                .chart h4 {{ color: #666; margin-bottom: 10px; font-size: 14px; }}
                .chart img {{ max-width: 100%; height: 200px; object-fit: contain; border: 1px solid #ddd; border-radius: 4px; }}
                .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin-bottom: 20px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Distributed ML Platform - Model Visualizations Dashboard</h1>
                <div class="summary">
                    <strong>Total Models:</strong> {len(model_names)} | 
                    <strong>Datasets:</strong> {len(datasets)} | 
                    <strong>Status:</strong> ✅ All systems operational
                </div>
                {models_html}
            </div>
        </body>
        </html>
        """
        
        return StreamingResponse(
            io.BytesIO(html_content.encode()),
            media_type="text/html"
        )

    # 9. VISUALIZATION(model) - HTML page showing both charts for a model
    @app.get("/visualization/{model_name}")
    async def visualization_model(model_name: str):
        """Get HTML page displaying both ROC curve and learning curve for a specific model"""        
        try:
            actor = ray.get_actor(model_name)
            
            # Verify model exists
            ray.get(actor.get_metrics.remote())
            
            # Return HTML page with embedded images
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Visualizations for {model_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                    h1 {{ color: #333; text-align: center; border-bottom: 3px solid #4CAF50; padding-bottom: 20px; }}
                    .chart-container {{ display: flex; flex-wrap: wrap; gap: 30px; justify-content: center; margin-top: 30px; }}
                    .chart {{ text-align: center; flex: 1; min-width: 400px; }}
                    .chart h2 {{ color: #666; margin-bottom: 20px; }}
                    .chart img {{ max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    .info {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Model Visualizations: {model_name}</h1>
                    <div class="info">
                        <strong>Dataset:</strong> {model_name.split('_')[-1] if '_' in model_name else 'Unknown'} | 
                        <strong>Model Type:</strong> {model_name.split('_')[0] if '_' in model_name else model_name}
                    </div>
                    <div class="chart-container">
                        <div class="chart">
                            <h2>ROC Curve</h2>
                            <img src="/visualization/{model_name}/roc" alt="ROC Curve for {model_name}">
                        </div>
                        <div class="chart">
                            <h2>Learning Curve</h2>
                            <img src="/visualization/{model_name}/learning_curve" alt="Learning Curve for {model_name}">
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            return StreamingResponse(
                io.BytesIO(html_content.encode()),
                media_type="text/html"
            )
            
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found: {str(e)}")

    # 10. DATASETS - List available datasets
    @app.get("/datasets")
    async def list_datasets():
        """List all available datasets based on trained models"""
        datasets = set()
        for model_name in model_names:
            # Extract dataset name from model name (format: ModelType_dataset)
            if '_' in model_name:
                dataset = model_name.split('_', 1)[1]  # Split only on first underscore
                datasets.add(dataset)
        return {'datasets': sorted(list(datasets))}
    
    # 11. MODELS by dataset - List models for specific dataset
    @app.get("/models/{dataset}")
    async def list_models_for_dataset(dataset: str):
        """List all models trained on a specific dataset"""
        dataset_models = [name for name in model_names if name.endswith(f"_{dataset}")]
        if not dataset_models:
            raise HTTPException(status_code=404, detail=f"No models found for dataset: {dataset}")
        return {'dataset': dataset, 'models': dataset_models}

    # Health check for all models
    @app.get("/models/health")
    async def models_health():
        """Check health of all model actors"""
        try:
            import ray.util
            healthy_models = []
            for actor_name in ray.util.list_named_actors():
                if any(dataset in actor_name for dataset in ['iris', 'wine', 'breast_cancer']):
                    try:
                        actor = ray.get_actor(actor_name)
                        ray.get(actor.get_name.remote(), timeout=2.0)
                        healthy_models.append(actor_name)
                    except Exception:
                        pass
            return {'healthy_models': healthy_models, 'count': len(healthy_models)}
        except Exception as e:
            return {'error': str(e), 'healthy_models': [], 'count': 0}

    # NEW ENDPOINTS FOR STREAMLIT WORKFLOW
    
    # File upload endpoint for distributed data ingestion
    @app.post("/upload", response_model=FileUploadResponse)
    async def upload_file(request: FileUploadRequest):
        """Upload and process a file in a distributed manner"""
        try:
            import pandas as pd
            import base64
            import io
            
            # Decode base64 content
            content_bytes = base64.b64decode(request.content)
            
            # Determine file type and read accordingly
            filename_lower = request.filename.lower()
            if filename_lower.endswith('.csv'):
                df = pd.read_csv(io.StringIO(content_bytes.decode('utf-8')))
            elif filename_lower.endswith('.json'):
                df = pd.read_json(io.StringIO(content_bytes.decode('utf-8')))
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {request.filename}")
            
            # Distribute data loading across Ray cluster
            from src.data.data_loader import DataLoaderActor
            
            # Create a data loader actor for this file
            loader_actor_name = f"data_loader_{request.filename.replace('.', '_')}"
            loader_actor = DataLoaderActor.options(name=loader_actor_name).remote()
            
            # Store the data in the distributed actor
            ray.get(loader_actor.load_data.remote(df.to_dict('records')))
            
            # Generate preview
            preview = df.head(5).to_dict('records')
            
            return FileUploadResponse(
                filename=request.filename,
                status="uploaded_and_distributed",
                rows=len(df),
                columns=list(df.columns),
                preview=preview
            )
            
        except Exception as e:
            logger.error(f"Error uploading file {request.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    @app.get("/uploaded_files")
    async def list_uploaded_files():
        """List all uploaded files stored in Ray cluster"""
        try:
            import ray.util
            uploaded_files = []
            for actor_name in ray.util.list_named_actors():
                if actor_name.startswith('data_loader_'):
                    filename = actor_name.replace('data_loader_', '').replace('_', '.')
                    try:
                        actor = ray.get_actor(actor_name)
                        data_info = ray.get(actor.get_data_info.remote())
                        uploaded_files.append({
                            'filename': filename,
                            'actor_name': actor_name,
                            'rows': data_info.get('rows', 0),
                            'columns': data_info.get('columns', [])
                        })
                    except Exception:
                        pass
            return {'uploaded_files': uploaded_files}
        except Exception as e:
            return {'error': str(e), 'uploaded_files': []}
    
    # Training endpoint (on-demand)
    @app.post("/train")
    async def train_models(request: Dict[str, Any]):
        """Start distributed training with user-provided datasets and configuration"""
        try:
            datasets = request.get('datasets', {})
            ml_tasks = request.get('ml_tasks', {})
            targets = request.get('targets', {})
            model_selections = request.get('model_selections', {})  # Optional model selection per dataset
            
            if not datasets or not ml_tasks or not targets:
                raise HTTPException(status_code=400, detail="Missing required fields: datasets, ml_tasks, targets")
            
            # Import training function
            from src.models.model_trainer import train_multiple_models, ModelActor
            import pandas as pd
            import numpy as np
            
            # Process each dataset
            results = {}
            for dataset_name, dataset_records in datasets.items():
                df = pd.DataFrame(dataset_records)
                task = ml_tasks[dataset_name]
                target = targets[dataset_name]
                selected_models = model_selections.get(dataset_name, [])
                
                # Convert task name to classification/regression
                is_classification = task == "Clasificación"
                
                # Split features and target
                if target not in df.columns:
                    raise HTTPException(status_code=400, detail=f"Target column '{target}' not found in dataset '{dataset_name}'")
                
                X = df.drop(target, axis=1)
                y = df[target]
                
                # Convert categorical features to numeric
                X = pd.get_dummies(X)
                
                # Train split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Define available models based on task type
                available_models = {}
                available_model_names = {}
                
                if is_classification:
                    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.svm import SVC
                    from sklearn.neighbors import KNeighborsClassifier
                    
                    available_models = {
                        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
                        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
                        "SVC": SVC(probability=True, random_state=42),
                        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5)
                    }
                else:
                    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
                    
                    available_models = {
                        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
                        "LinearRegression": LinearRegression(),
                        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
                        "Ridge": Ridge(random_state=42),
                        "Lasso": Lasso(random_state=42),
                        "ElasticNet": ElasticNet(random_state=42)
                    }
                
                # Filter models based on user selection (if any)
                if selected_models:
                    models = [available_models[model_name] for model_name in selected_models if model_name in available_models]
                    model_names = [f"{model_name}_{dataset_name}" for model_name in selected_models if model_name in available_models]
                else:
                    # Use all available models if none selected
                    models = list(available_models.values())
                    model_names = [f"{model_name}_{dataset_name}" for model_name in available_models.keys()]
                
                if not models:
                    continue  # Skip this dataset if no valid models selected
                
                # Start distributed training
                trained_models, training_metrics = train_multiple_models(
                    models, X_train, y_train, X_test, y_test, model_names
                )
                
                # Create Ray actors (NO disk storage - purely distributed)
                for model_name, model in trained_models.items():
                    # Create Ray actor with model in distributed memory
                    try:
                        actor = ModelActor.options(name=model_name, lifetime="detached").remote(model, model_name)
                        actor.set_metrics.remote(training_metrics.get(model_name, {}))
                        actor.set_training_data.remote(X_train, y_train, X_test, y_test)
                        logger.info(f"Created Ray actor for model: {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to create actor for {model_name}: {e}")
                
                results[dataset_name] = {
                    "models_trained": len(trained_models),
                    "metrics": training_metrics,
                    "model_names": list(trained_models.keys())
                }
            
            return {"status": "training_completed", "results": results}
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/train_uploaded")
    async def train_uploaded_files(request: Dict[str, Any]):
        """Start distributed training with uploaded files from Ray actors"""
        try:
            file_configs = request.get('file_configs', {})  # {filename: {task, target, models}}
            
            if not file_configs:
                raise HTTPException(status_code=400, detail="Missing required field: file_configs")
            
            # Import training function
            from src.models.model_trainer import train_multiple_models, ModelActor
            from src.data.data_loader import DataLoaderActor
            import pandas as pd
            import numpy as np
            
            # Process each uploaded file
            results = {}
            for filename, config in file_configs.items():
                task = config.get('task')
                target = config.get('target')
                selected_models = config.get('models', [])
                
                if not task or not target:
                    continue  # Skip files without task/target
                
                # Get data from Ray actor
                actor_name = f"data_loader_{filename.replace('.', '_')}"
                try:
                    loader_actor = ray.get_actor(actor_name)
                    dataset_records = ray.get(loader_actor.get_data.remote())
                    
                    if not dataset_records:
                        continue  # Skip if no data
                    
                    df = pd.DataFrame(dataset_records)
                    
                except Exception as e:
                    logger.error(f"Failed to get data for {filename}: {e}")
                    continue
                
                # Convert task name to classification/regression
                is_classification = task == "Clasificación"
                
                # Split features and target
                if target not in df.columns:
                    logger.error(f"Target column '{target}' not found in dataset '{filename}'")
                    continue
                
                X = df.drop(target, axis=1)
                y = df[target]
                
                # Convert categorical features to numeric
                X = pd.get_dummies(X)
                
                # Train split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Define available models based on task type
                available_models = {}
                
                if is_classification:
                    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.svm import SVC
                    from sklearn.neighbors import KNeighborsClassifier
                    
                    available_models = {
                        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
                        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
                        "SVC": SVC(probability=True, random_state=42),
                        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5)
                    }
                else:
                    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
                    
                    available_models = {
                        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
                        "LinearRegression": LinearRegression(),
                        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
                        "Ridge": Ridge(random_state=42),
                        "Lasso": Lasso(random_state=42),
                        "ElasticNet": ElasticNet(random_state=42)
                    }
                
                # Filter models based on user selection (if any)
                dataset_name = filename.replace('.csv', '').replace('.json', '')
                if selected_models:
                    models = [available_models[model_name] for model_name in selected_models if model_name in available_models]
                    model_names = [f"{model_name}_{dataset_name}" for model_name in selected_models if model_name in available_models]
                else:
                    # Use all available models if none selected
                    models = list(available_models.values())
                    model_names = [f"{model_name}_{dataset_name}" for model_name in available_models.keys()]
                
                if not models:
                    continue  # Skip this dataset if no valid models selected
                
                # Start distributed training
                trained_models, training_metrics = train_multiple_models(
                    models, X_train, y_train, X_test, y_test, model_names
                )
                
                # Create Ray actors (NO disk storage - purely distributed)
                for model_name, model in trained_models.items():
                    # Create Ray actor with model in distributed memory
                    try:
                        actor = ModelActor.options(name=model_name, lifetime="detached").remote(model, model_name)
                        actor.set_metrics.remote(training_metrics.get(model_name, {}))
                        actor.set_training_data.remote(X_train, y_train, X_test, y_test)
                        logger.info(f"Created Ray actor for model: {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to create actor for {model_name}: {e}")
                
                results[filename] = {
                    "models_trained": len(trained_models),
                    "metrics": training_metrics,
                    "model_names": list(trained_models.keys())
                }
            
            return {"status": "training_completed", "results": results}
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Cluster management endpoints
    @app.get("/cluster/status")
    async def cluster_status():
        """Get Ray cluster status with better error handling"""
        try:
            import ray
            
            # Check if Ray is initialized
            if not ray.is_initialized():
                return {
                    "error": "Ray has not been started yet. You can start Ray with 'ray.init()'.",
                    "status": "not_initialized",
                    "suggestion": "Ray cluster is starting up or not properly initialized"
                }
            
            # Get cluster information
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            nodes = ray.nodes()
            
            # Additional cluster metrics
            active_nodes = len([node for node in nodes if node['Alive']])
            total_cpus = cluster_resources.get('CPU', 0)
            available_cpus = available_resources.get('CPU', 0)
            total_memory = cluster_resources.get('memory', 0)
            available_memory = available_resources.get('memory', 0)
            
            return {
                "status": "healthy",
                "cluster_resources": cluster_resources,
                "available_resources": available_resources,
                "nodes": len(nodes),
                "active_nodes": active_nodes,
                "node_details": nodes,
                "summary": {
                    "total_cpus": total_cpus,
                    "available_cpus": available_cpus,
                    "cpu_utilization": round((total_cpus - available_cpus) / total_cpus * 100, 2) if total_cpus > 0 else 0,
                    "total_memory_gb": round(total_memory / 1e9, 2),
                    "available_memory_gb": round(available_memory / 1e9, 2),
                    "memory_utilization": round((total_memory - available_memory) / total_memory * 100, 2) if total_memory > 0 else 0
                }
            }
        except Exception as e:
            logger.error(f"Error getting cluster status: {e}")
            return {
                "error": str(e), 
                "status": "error",
                "suggestion": "Check if Ray cluster is properly running and accessible"
            }
    
    import os
    import subprocess
    import asyncio
    
    def get_docker_compose_path():
        """Get the path to docker-compose.yml file"""
        # Navigate up from src/serving/api.py to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        return project_root
    
    async def get_current_worker_count():
        """Get current number of Ray worker containers"""
        try:
            project_root = get_docker_compose_path()
            result = subprocess.run(
                ["docker", "compose", "ps", "--format", "json"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            import json
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    containers.append(json.loads(line))
            
            # Count containers with ray_worker in service name
            worker_count = len([c for c in containers if 'ray_worker' in c.get('Service', '') and c.get('State') == 'running'])
            return worker_count
        except Exception as e:
            logger.error(f"Error getting worker count: {e}")
            return None
    
    async def scale_ray_workers(target_count):
        """Scale Ray workers to target count using docker compose"""
        try:
            project_root = get_docker_compose_path()
            
            # Use docker compose scale command
            process = await asyncio.create_subprocess_exec(
                "docker", "compose", "up", "-d", "--scale", f"ray_worker={target_count}",
                cwd=project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown docker compose error"
                logger.error(f"Docker compose scaling failed: {error_msg}")
                return False, error_msg
            
            # Wait for Ray to register changes
            await asyncio.sleep(5)
            
            # Verify the scaling worked by checking Ray cluster
            import ray
            for attempt in range(15):  # Wait up to 15 seconds
                try:
                    nodes = ray.nodes()
                    alive_nodes = [n for n in nodes if n['Alive']]
                    # Head node + workers should equal target_count + 1
                    if len(alive_nodes) >= target_count + 1:
                        break
                except Exception:
                    pass
                await asyncio.sleep(1)
            
            return True, f"Successfully scaled to {target_count} workers"
            
        except Exception as e:
            logger.error(f"Error scaling workers: {e}")
            return False, str(e)
    
    @app.post("/cluster/add_worker")
    async def add_worker():
        """Add a new Ray worker container to the cluster (REAL Docker Compose scaling)"""
        try:
            current_count = await get_current_worker_count()
            if current_count is None:
                return {"error": "Could not determine current worker count", "success": False}
            
            new_count = current_count + 1
            logger.info(f"Adding Ray worker: {current_count} -> {new_count}")
            
            success, message = await scale_ray_workers(new_count)
            
            if success:
                # Verify Ray cluster sees the new worker
                import ray
                if ray.is_initialized():
                    nodes = ray.nodes()
                    alive_workers = len([n for n in nodes if n['Alive']]) - 1  # Subtract head node
                    return {
                        "success": True,
                        "message": message,
                        "previous_workers": current_count,
                        "current_workers": new_count,
                        "ray_cluster_workers": alive_workers
                    }
                else:
                    return {"success": True, "message": message, "note": "Ray not initialized to verify"}
            else:
                return {"error": message, "success": False}
                
        except Exception as e:
            logger.error(f"Error adding Ray worker: {e}")
            return {"error": str(e), "success": False}
    
    @app.post("/cluster/remove_worker")
    async def remove_worker():
        """Remove a Ray worker container from the cluster (REAL Docker Compose scaling)"""
        try:
            current_count = await get_current_worker_count()
            if current_count is None:
                return {"error": "Could not determine current worker count", "success": False}
            
            if current_count <= 1:
                return {"error": "Cannot remove the last worker - minimum 1 worker required", "success": False}
            
            new_count = current_count - 1
            logger.info(f"Removing Ray worker: {current_count} -> {new_count}")
            
            success, message = await scale_ray_workers(new_count)
            
            if success:
                # Verify Ray cluster sees the change
                import ray
                if ray.is_initialized():
                    nodes = ray.nodes()
                    alive_workers = len([n for n in nodes if n['Alive']]) - 1  # Subtract head node
                    return {
                        "success": True,
                        "message": message,
                        "previous_workers": current_count,
                        "current_workers": new_count,
                        "ray_cluster_workers": alive_workers
                    }
                else:
                    return {"success": True, "message": message, "note": "Ray not initialized to verify"}
            else:
                return {"error": message, "success": False}
                
        except Exception as e:
            logger.error(f"Error removing Ray worker: {e}")
            return {"error": str(e), "success": False}

    @app.post("/cluster/pause_worker/{worker_number}")
    async def pause_worker(worker_number: int):
        """Pause a specific Ray worker container by stopping it"""
        try:
            project_root = get_docker_compose_path()
            
            # Get all ray_worker containers
            result = subprocess.run(
                ["docker", "compose", "ps", "--format", "json"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            import json
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    containers.append(json.loads(line))
            
            # Find ray_worker containers sorted by name
            ray_workers = sorted([c for c in containers if 'ray_worker' in c.get('Service', '')], 
                               key=lambda x: x.get('Name', ''))
            
            if worker_number < 1 or worker_number > len(ray_workers):
                return {"error": f"Worker {worker_number} not found. Available workers: 1-{len(ray_workers)}", "success": False}
            
            # Get the specific worker container
            target_worker = ray_workers[worker_number - 1]
            container_name = target_worker.get('Name')
            
            # Stop the specific container
            process = await asyncio.create_subprocess_exec(
                "docker", "stop", container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown docker error"
                logger.error(f"Failed to pause worker {worker_number}: {error_msg}")
                return {"error": f"Failed to pause worker {worker_number}: {error_msg}", "success": False}
            
            return {
                "success": True,
                "message": f"Worker {worker_number} ({container_name}) paused successfully",
                "worker_number": worker_number,
                "container_name": container_name
            }
            
        except Exception as e:
            logger.error(f"Error pausing worker {worker_number}: {e}")
            return {"error": str(e), "success": False}
    
    @app.post("/cluster/delete_worker/{worker_number}")
    async def delete_worker(worker_number: int):
        """Delete a specific Ray worker container completely"""
        try:
            project_root = get_docker_compose_path()
            
            # Get all ray_worker containers
            result = subprocess.run(
                ["docker", "compose", "ps", "--format", "json"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            import json
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    containers.append(json.loads(line))
            
            # Find ray_worker containers sorted by name
            ray_workers = sorted([c for c in containers if 'ray_worker' in c.get('Service', '')], 
                               key=lambda x: x.get('Name', ''))
            
            if worker_number < 1 or worker_number > len(ray_workers):
                return {"error": f"Worker {worker_number} not found. Available workers: 1-{len(ray_workers)}", "success": False}
            
            # Get the specific worker container
            target_worker = ray_workers[worker_number - 1]
            container_name = target_worker.get('Name')
            
            # Stop and remove the specific container
            stop_process = await asyncio.create_subprocess_exec(
                "docker", "stop", container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await stop_process.communicate()
            
            remove_process = await asyncio.create_subprocess_exec(
                "docker", "rm", container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await remove_process.communicate()
            
            if remove_process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown docker error"
                logger.error(f"Failed to delete worker {worker_number}: {error_msg}")
                return {"error": f"Failed to delete worker {worker_number}: {error_msg}", "success": False}
            
            return {
                "success": True,
                "message": f"Worker {worker_number} ({container_name}) deleted successfully",
                "worker_number": worker_number,
                "container_name": container_name
            }
            
        except Exception as e:
            logger.error(f"Error deleting worker {worker_number}: {e}")
            return {"error": str(e), "success": False}
    
    @app.get("/cluster/workers")
    async def get_workers():
        """Get detailed information about all Ray worker containers"""
        try:
            project_root = get_docker_compose_path()
            
            # Get all containers
            result = subprocess.run(
                ["docker", "compose", "ps", "--format", "json"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            import json
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    containers.append(json.loads(line))
            
            # Find ray_worker containers sorted by name
            ray_workers = sorted([c for c in containers if 'ray_worker' in c.get('Service', '')], 
                               key=lambda x: x.get('Name', ''))
            
            # Get detailed info for each worker
            worker_details = []
            for i, worker in enumerate(ray_workers, 1):
                # Get container stats
                stats_result = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format", "table {{.CPUPerc}}\t{{.MemUsage}}", worker.get('Name')],
                    capture_output=True,
                    text=True
                )
                
                cpu_usage = "N/A"
                memory_usage = "N/A"
                if stats_result.returncode == 0 and stats_result.stdout:
                    lines = stats_result.stdout.strip().split('\n')
                    if len(lines) > 1:  # Skip header
                        parts = lines[1].split('\t')
                        if len(parts) >= 2:
                            cpu_usage = parts[0]
                            memory_usage = parts[1]
                
                worker_details.append({
                    "number": i,
                    "name": worker.get('Name'),
                    "status": worker.get('State'),
                    "service": worker.get('Service'),
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage
                })
            
            return {
                "success": True,
                "workers": worker_details,
                "total_workers": len(worker_details)
            }
            
        except Exception as e:
            logger.error(f"Error getting worker details: {e}")
            return {"error": str(e), "success": False}

    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)}
        )

    return app, predictor