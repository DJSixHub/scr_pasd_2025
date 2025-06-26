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
                # Define classification models to train
                models = [
                    RandomForestClassifier(n_estimators=100),
                    GradientBoostingClassifier(),
                    LogisticRegression(max_iter=1000),
                    KNeighborsClassifier(n_neighbors=5),
                    SVC(probability=True, gamma='auto'),
                    DecisionTreeClassifier(),
                    AdaBoostClassifier(),
                    ExtraTreesClassifier(),
                    MLPClassifier(max_iter=500, hidden_layer_sizes=(50,50)),
                    GaussianNB(),
                    SGDClassifier(max_iter=1000)
                ]
                
                model_names = [
                    f"RandomForest_ds{i+1}",
                    f"GradientBoosting_ds{i+1}",
                    f"LogisticRegression_ds{i+1}",
                    f"KNN_ds{i+1}",
                    f"SVC_ds{i+1}",
                    f"DecisionTree_ds{i+1}",
                    f"AdaBoost_ds{i+1}",
                    f"ExtraTrees_ds{i+1}",
                    f"NeuralNet_ds{i+1}",
                    f"NaiveBayes_ds{i+1}",
                    f"SGD_ds{i+1}"
                ]
            else:
                # Define regression models to train
                models = [
                    Ridge(alpha=1.0),
                    Lasso(alpha=0.1),
                    ElasticNet(alpha=0.1, l1_ratio=0.5),
                    SVR(gamma='auto'),
                    KNeighborsRegressor(n_neighbors=5),
                    DecisionTreeRegressor(),
                    RandomForestClassifier(n_estimators=100),
                    GradientBoostingClassifier(),
                    MLPRegressor(max_iter=500, hidden_layer_sizes=(50,50)),
                    SGDRegressor(max_iter=1000)
                ]
                
                model_names = [
                    f"Ridge_ds{i+1}",
                    f"Lasso_ds{i+1}",
                    f"ElasticNet_ds{i+1}",
                    f"SVR_ds{i+1}",
                    f"KNNRegressor_ds{i+1}",
                    f"DecisionTreeRegressor_ds{i+1}",
                    f"RandomForestRegressor_ds{i+1}",
                    f"GradientBoostingRegressor_ds{i+1}",
                    f"NeuralNetRegressor_ds{i+1}",
                    f"SGDRegressor_ds{i+1}"
                ]
            
            # Train models with fault tolerance
            dataset_models, dataset_metrics = train_multiple_models(
                models, X_train, y_train, X_test, y_test, model_names,
                checkpoint_dir=checkpoint_dir if args.failure_tolerant else None,
                max_retries=3 if args.failure_tolerant else 1,
                timeout=600
            )
            
            # Save models with redundancy if requested
            model_dir = os.path.join(args.output_dir, 'models')
            if args.ray_redundancy > 1:
                save_models(dataset_models, model_dir, replicas=args.ray_redundancy)
            else:
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
    
    # Serving phase
    if args.mode in ['serve', 'all']:
        logger.info("Starting serving phase")
        
        if not trained_models:
            logger.info("No models trained in this session, loading from disk")
            
            # Build search paths for models with redundancy support
            model_dirs = [args.model_dir]
            
            if args.model_backup_dirs:
                model_dirs.extend(args.model_backup_dirs)
            
            if args.failure_tolerant and args.ray_redundancy > 1:
                # Add potential model redundancy locations
                for i in range(1, args.ray_redundancy):
                    backup_dir = f"{args.model_dir}_replica_{i}"
                    if backup_dir not in model_dirs:
                        model_dirs.append(backup_dir)
                
                # Add additional potential locations
                model_dirs.extend([
                    '/app/output/models', 
                    '/app/output_backup/models'
                ])
            
            # Gather all potential model files from all directories
            model_files = []
            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    for file in os.listdir(model_dir):
                        if file.endswith('.joblib'):
                            model_files.append(os.path.join(model_dir, file))
            
            if not model_files:
                logger.error("No models found for serving")
                sys.exit(1)
            
            # Load models with redundancy support
            if args.failure_tolerant:
                trained_models = load_models(model_files, verify_integrity=args.verify_models)
            else:
                trained_models = load_models(model_files)
                
            if not trained_models:
                logger.error("Failed to load any models for serving")
                sys.exit(1)
            
            logger.info(f"Loaded {len(trained_models)} models for serving")
        
        # Create and start the API server
        logger.info(f"Starting API server on {args.host}:{args.port}")
        create_api(trained_models, host=args.host, port=args.port)

    # Shutdown Ray
    ray.shutdown()
    logger.info("Shut down Ray")
    
if __name__ == "__main__":
    main()
