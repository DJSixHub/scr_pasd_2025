"""
Main module for the distributed ML platform
"""
import os
import argparse
import logging
import time
import sys
import ray
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB

from ray import serve

from src.utils.ray_utils import initialize_ray, get_cluster_status
from src.data.data_loader import load_and_preprocess_data
from src.models.model_trainer import train_multiple_models, ModelActor
from src.serving.api import create_app
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
    parser.add_argument('--min-nodes', type=int, help='Minimum number of Ray nodes required', default=1)
    parser.add_argument('--failure-tolerant', action='store_true', help='Enable failure tolerance features')
    parser.add_argument('--ray-redundancy', type=int, help='Redundancy level for data and models (1=no redundancy)', default=1)
    
    # Operation mode
    parser.add_argument('--mode', type=str, choices=['train', 'serve', 'all'], 
                      help='Operation mode: train models, serve models, or both', default='all')
    
    # Training settings
    parser.add_argument('--data', type=str, nargs='+', help='Paths to datasets for training')
    parser.add_argument('--data-dirs', type=str, nargs='+', help='Additional directories to search for redundant data')
    parser.add_argument('--target', type=str, help='Target column name for training')
    parser.add_argument('--test-size', type=float, help='Test size for train/test split', default=0.2)
    parser.add_argument('--scale', action='store_true', help='Scale features before training')
    parser.add_argument('--output-dir', type=str, help='Directory to save models and plots', default='output')
    parser.add_argument('--checkpoint-dir', type=str, help='Directory for model checkpoints during training')
    parser.add_argument('--verify-data', action='store_true', help='Verify data integrity before loading')
    
    # Serving settings
    parser.add_argument('--model-dir', type=str, help='Directory containing trained models', default='output/models')
    parser.add_argument('--model-backup-dirs', type=str, nargs='+', help='Backup directories for model redundancy')
    parser.add_argument('--host', type=str, help='Host to bind the API server to', default='0.0.0.0')
    parser.add_argument('--port', type=int, help='Port to bind the API server to', default=8000)
    parser.add_argument('--verify-models', action='store_true', help='Verify model integrity before loading')
    
    # Visualization settings
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--show-plots', action='store_true', help='Show plots during execution')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_args()
    
    # Initialize Ray
    logger.info("Initializing Ray...")
    if args.failure_tolerant:
        logger.info("Running in failure-tolerant mode")
        from src.utils.ray_utils import RayFailoverManager, check_cluster_health
        
        # Create a failover manager with potential cluster addresses
        failover_manager = RayFailoverManager(
            primary_address=args.address,
            secondary_addresses=["auto", "localhost:6379"]
        )
        
        # Connect with failover support
        if not failover_manager.connect(max_retries=3):
            logger.error("Failed to connect to any Ray cluster, exiting")
            sys.exit(1)
            
        # Check cluster health if in distributed mode
        if not args.local:
            logger.info("Checking cluster health...")
            if not check_cluster_health(min_nodes=args.min_nodes):
                logger.warning(f"Cluster does not meet minimum requirements of {args.min_nodes} nodes")
                if args.failure_tolerant:
                    logger.info("Continuing in degraded mode due to failure tolerance setting")
                else:
                    logger.error("Exiting due to insufficient cluster resources")
                    sys.exit(1)
    else:
        # Standard Ray initialization
        if not initialize_ray(args.address, args.local):
            logger.error("Failed to initialize Ray, exiting")
            sys.exit(1)
    
    logger.info("Ray initialized successfully")
    
    # Track all trained models and their metrics
    trained_models = {}
    model_actors = {}
    training_metrics = {}
    
    # Training phase
    if args.mode in ['train', 'all']:
        logger.info("Starting training phase")
        
        if not args.data:
            logger.error("No datasets provided for training. Use --data to specify dataset paths.")
            sys.exit(1)
            
        logger.info(f"Loading and preprocessing {len(args.data)} datasets")
        
        # Build comprehensive search directories for data redundancy
        data_dirs = []
        if args.data_dirs:
            data_dirs.extend(args.data_dirs)
            
        if args.failure_tolerant:
            # Add potential data redundancy locations
            data_dirs.extend([
                '/app/data',
                '/tmp/ray_data/data',
                './data_backup',
                '/app/data_backup'
            ])
            
        # Create checkpoint directory if specified
        checkpoint_dir = args.checkpoint_dir
        if args.failure_tolerant and not checkpoint_dir:
            checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load and preprocess data with redundancy support
        processed_datasets = load_and_preprocess_data(
            args.data,
            target_col=args.target,
            test_size=args.test_size,
            scale=args.scale,
            data_dirs=data_dirs if args.ray_redundancy > 1 else []
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
            
            # Check if it's a classification or regression task
            is_classification = len(np.unique(y_train)) < 10  # Assuming classification if fewer classes
            
            if is_classification:
                # Define classification models to train (3 models for fast testing)
                models = [
                    RandomForestClassifier(n_estimators=100),
                    LogisticRegression(max_iter=1000),
                    GradientBoostingClassifier()
                ]
                
                model_names = [
                    f"RandomForest_ds{i+1}",
                    f"LogisticRegression_ds{i+1}",
                    f"GradientBoosting_ds{i+1}"
                ]
            else:
                # Define regression models to train (3 models for fast testing)
                models = [
                    Ridge(alpha=1.0),
                    ElasticNet(alpha=0.1, l1_ratio=0.5),
                    DecisionTreeRegressor()
                ]
                
                model_names = [
                    f"Ridge_ds{i+1}",
                    f"ElasticNet_ds{i+1}",
                    f"DecisionTreeRegressor_ds{i+1}"
                ]
        
            # Train models with fault tolerance
            dataset_models, dataset_metrics = train_multiple_models(
                models, X_train, y_train, X_test, y_test, model_names,
                max_retries=3 if args.failure_tolerant else 1,
                timeout=600
            )
            
            # Deploy each trained model as a Ray actor
            for model_name, model in dataset_models.items():
                actor = ModelActor.options(name=model_name, lifetime="detached").remote(model, model_name)
                actor.set_metrics.remote(dataset_metrics.get(model_name, {}))
                model_actors[model_name] = actor
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
            # Check if metrics are dictionaries (not scalar values)
            has_detailed_metrics = all(isinstance(metrics, dict) for metrics in training_metrics.values())
            
            if has_detailed_metrics:
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
    
    # Serving phase
    if args.mode in ['serve', 'all']:
        logger.info("Starting serving phase")
        # Discover all Ray model actors by name
        import ray
        if not ray.is_initialized():
            ray.init()
        # List all named actors (models)
        model_names = []
        try:
            model_names = [a for a in ray.util.list_named_actors() if not a.startswith("__")]
        except Exception:
            pass
        if not model_names:
            logger.error("No distributed model actors found for serving")
            sys.exit(1)
        logger.info(f"Found {len(model_names)} distributed model actors for serving: {model_names}")
        # Pass only model names to the API, which will use Ray to route requests
        app, _ = create_app(model_names)
        
        # Run FastAPI directly with Uvicorn instead of Ray Serve for simplicity
        import uvicorn
        logger.info(f"Starting API server at http://{args.host}:{args.port}")
        
        # Run the server in a way that allows graceful shutdown
        config = uvicorn.Config(app=app, host=args.host, port=args.port, log_level="info")
        server = uvicorn.Server(config)
        
        try:
            server.run()
        except KeyboardInterrupt:
            logger.info("Shutting down API server...")
        finally:
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Shut down Ray")
    
if __name__ == "__main__":
    main()
