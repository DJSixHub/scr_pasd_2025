"""
Main module for the distributed ML platform
"""
import os
import argparse
import logging
import time
import json
import sys
import ray
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.utils.ray_utils import initialize_ray, get_cluster_status
from src.data.data_loader import load_and_preprocess_data
from src.models.model_trainer import train_multiple_models, save_models, load_models
from src.serving.api import create_api
from src.visualization.visualizer import plot_training_metrics, plot_model_comparison, plot_inference_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_platform.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Distributed ML Platform')
    
    # Ray connection settings
    parser.add_argument('--address', type=str, help='Ray cluster address (e.g., auto, localhost:6379)', default=None)
    parser.add_argument('--local', action='store_true', help='Force Ray to run in local mode')
    
    # Operation mode
    parser.add_argument('--mode', type=str, choices=['train', 'serve', 'all'], 
                      help='Operation mode: train models, serve models, or both', default='all')
    
    # Training settings
    parser.add_argument('--data', type=str, nargs='+', help='Paths to datasets for training')
    parser.add_argument('--target', type=str, help='Target column name for training')
    parser.add_argument('--test-size', type=float, help='Test size for train/test split', default=0.2)
    parser.add_argument('--scale', action='store_true', help='Scale features before training')
    parser.add_argument('--output-dir', type=str, help='Directory to save models and plots', default='output')
    
    # Serving settings
    parser.add_argument('--model-dir', type=str, help='Directory containing trained models', default='output/models')
    parser.add_argument('--host', type=str, help='Host to bind the API server to', default='0.0.0.0')
    parser.add_argument('--port', type=int, help='Port to bind the API server to', default=8000)
    
    # Visualization settings
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--show-plots', action='store_true', help='Show plots during execution')
    
    return parser.parse_args()

def main():
    """Main entry point for the distributed ML platform"""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    
    # Initialize Ray
    initialize_ray(address=args.address, local=args.local)
    logger.info(f"Ray cluster status: {get_cluster_status()}")
    
    trained_models = {}
    training_metrics = {}
    
    # Training mode
    if args.mode in ['train', 'all']:
        if not args.data:
            logger.error("No datasets provided for training. Use --data to specify dataset paths.")
            sys.exit(1)
            
        logger.info(f"Loading and preprocessing {len(args.data)} datasets")
        processed_datasets = load_and_preprocess_data(
            args.data,
            target_col=args.target,
            test_size=args.test_size,
            scale=args.scale
        )
        
        if not processed_datasets:
            logger.error("Failed to load and preprocess datasets.")
            sys.exit(1)
        
        # Train models for each dataset
        for i, dataset in enumerate(processed_datasets):
            if dataset is None or len(dataset) < 4:
                logger.warning(f"Dataset {i} is invalid, skipping...")
                continue
                
            X_train, X_test, y_train, y_test = dataset
            
            logger.info(f"Training models for dataset {i+1}/{len(processed_datasets)}")
            
            # Define models to train
            models = [
                RandomForestClassifier(),
                GradientBoostingClassifier(),
                LogisticRegression(max_iter=1000),
                KNeighborsClassifier(),
                SVC(probability=True)
            ]
            
            model_names = [
                f"RandomForest_ds{i+1}",
                f"GradientBoosting_ds{i+1}",
                f"LogisticRegression_ds{i+1}",
                f"KNN_ds{i+1}",
                f"SVC_ds{i+1}"
            ]
            
            # Train models
            dataset_models, dataset_metrics = train_multiple_models(
                models, X_train, y_train, X_test, y_test, model_names
            )
            
            # Save models
            model_dir = os.path.join(args.output_dir, 'models')
            save_models(dataset_models, model_dir)
            
            # Update global dictionaries
            trained_models.update(dataset_models)
            training_metrics.update(dataset_metrics)
            
            # Generate visualizations for this dataset
            if not args.no_plots:
                logger.info(f"Generating visualizations for dataset {i+1}")
                plot_dir = os.path.join(args.output_dir, 'plots', f'dataset_{i+1}')
                plot_training_metrics(
                    dataset_metrics, 
                    output_dir=plot_dir,
                    save=True,
                    show=args.show_plots
                )
        
        # Generate overall visualizations
        if not args.no_plots and len(training_metrics) > 0:
            logger.info("Generating overall visualizations")
            plot_training_metrics(
                training_metrics, 
                output_dir=os.path.join(args.output_dir, 'plots'),
                save=True,
                show=args.show_plots
            )
            
            # Generate comparison plots for common metrics
            common_metrics = ['accuracy', 'precision', 'recall', 'f1']
            for metric in common_metrics:
                if all(metric in metrics for metrics in training_metrics.values()):
                    plot_model_comparison(
                        training_metrics,
                        metric_name=metric,
                        output_dir=os.path.join(args.output_dir, 'plots'),
                        save=True,
                        show=args.show_plots
                    )
    
    # Serving mode
    if args.mode in ['serve', 'all']:
        if not trained_models and args.mode == 'all':
            # Load models from disk for serving
            logger.info(f"Loading models from {args.model_dir} for serving")
            model_files = [os.path.join(args.model_dir, f) for f in os.listdir(args.model_dir) 
                          if f.endswith('.joblib')]
            trained_models = load_models(model_files)
        
        if trained_models:
            logger.info(f"Starting model serving API with {len(trained_models)} models")
            api = create_api(trained_models, host=args.host, port=args.port)
            
            try:
                # Keep the main thread alive
                while True:
                    time.sleep(10)
                    
                    # Generate inference visualizations
                    if not args.no_plots:
                        try:
                            # Get metrics from the API
                            metrics_response = api.app.test_client().get('/metrics')
                            metrics_data = json.loads(metrics_response.data)
                            
                            request_counts = metrics_data.get('request_counts', {})
                            latencies = metrics_data.get('average_latency_ms', {})
                            
                            if request_counts and latencies:
                                plot_inference_metrics(
                                    request_counts,
                                    latencies,
                                    output_dir=os.path.join(args.output_dir, 'plots', 'inference'),
                                    save=True,
                                    show=False
                                )
                        except Exception as e:
                            logger.error(f"Error generating inference visualizations: {e}")
            
            except KeyboardInterrupt:
                logger.info("Shutting down API...")
                api.stop()
        else:
            logger.error("No models available for serving.")
            sys.exit(1)
    
    # Shutdown Ray
    ray.shutdown()
    logger.info("Shut down Ray")
    
if __name__ == "__main__":
    main()
