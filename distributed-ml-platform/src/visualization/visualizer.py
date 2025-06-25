"""
Visualization utilities for monitoring models and system performance
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

logger = logging.getLogger(__name__)

def plot_training_metrics(metrics_dict, output_dir='plots', save=True, show=True):
    """
    Plot training metrics for multiple models
    
    Args:
        metrics_dict (dict): Dictionary mapping model names to metrics dictionaries
        output_dir (str): Directory to save plots
        save (bool): Whether to save the plots to disk
        show (bool): Whether to show the plots
        
    Returns:
        list: Paths to saved plot files if save=True
    """
    if not metrics_dict:
        logger.warning("No metrics provided for visualization")
        return []
        
    saved_paths = []
    
    # Create output directory
    if save:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine which metrics are available
    all_metric_keys = set()
    for model_metrics in metrics_dict.values():
        all_metric_keys.update(model_metrics.keys())
    
    # Remove training_time for separate plot
    metric_keys = [key for key in all_metric_keys if key != 'training_time']
    
    # Plot performance metrics
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics_dict))
    width = 0.8 / len(metric_keys)
    
    for i, metric_name in enumerate(metric_keys):
        values = [metrics.get(metric_name, 0) for model, metrics in metrics_dict.items()]
        plt.bar(x + i * width, values, width, label=metric_name)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')
    plt.xticks(x + width * (len(metric_keys) - 1) / 2, metrics_dict.keys(), rotation=45)
    plt.legend()
    plt.tight_layout()
    
    if save:
        file_path = os.path.join(output_dir, 'model_performance_metrics.png')
        plt.savefig(file_path)
        saved_paths.append(file_path)
        logger.info(f"Saved performance metrics plot to {file_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Plot training time
    plt.figure(figsize=(10, 6))
    training_times = [metrics.get('training_time', 0) for model, metrics in metrics_dict.items()]
    
    plt.bar(metrics_dict.keys(), training_times, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.title('Model Training Times')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save:
        file_path = os.path.join(output_dir, 'training_times.png')
        plt.savefig(file_path)
        saved_paths.append(file_path)
        logger.info(f"Saved training times plot to {file_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return saved_paths

def plot_model_comparison(metrics_dict, metric_name='accuracy', output_dir='plots', save=True, show=True):
    """
    Plot comparison of a specific metric across models
    
    Args:
        metrics_dict (dict): Dictionary mapping model names to metrics dictionaries
        metric_name (str): Name of the metric to compare
        output_dir (str): Directory to save plots
        save (bool): Whether to save the plots to disk
        show (bool): Whether to show the plots
        
    Returns:
        str or None: Path to saved plot file if save=True
    """
    if not metrics_dict:
        logger.warning("No metrics provided for visualization")
        return None
        
    # Create output directory
    if save:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract metric values
    models = list(metrics_dict.keys())
    values = [metrics.get(metric_name, 0) for model, metrics in metrics_dict.items()]
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=values)
    plt.xlabel('Models')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'Model Comparison by {metric_name.capitalize()}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    saved_path = None
    if save:
        file_path = os.path.join(output_dir, f'model_comparison_{metric_name}.png')
        plt.savefig(file_path)
        saved_path = file_path
        logger.info(f"Saved model comparison plot to {file_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return saved_path

def plot_inference_metrics(request_counts, latencies, output_dir='plots', save=True, show=True):
    """
    Plot inference metrics for models in production
    
    Args:
        request_counts (dict): Dictionary mapping model names to request counts
        latencies (dict): Dictionary mapping model names to average latencies
        output_dir (str): Directory to save plots
        save (bool): Whether to save the plots to disk
        show (bool): Whether to show the plots
        
    Returns:
        list: Paths to saved plot files if save=True
    """
    saved_paths = []
    
    # Create output directory
    if save:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot request counts
    plt.figure(figsize=(10, 6))
    plt.bar(request_counts.keys(), request_counts.values(), color='lightblue')
    plt.xlabel('Models')
    plt.ylabel('Number of Requests')
    plt.title('Model Request Counts')
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    
    if save:
        file_path = os.path.join(output_dir, 'request_counts.png')
        plt.savefig(file_path)
        saved_paths.append(file_path)
        logger.info(f"Saved request counts plot to {file_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Plot latencies
    plt.figure(figsize=(10, 6))
    plt.bar(latencies.keys(), latencies.values(), color='salmon')
    plt.xlabel('Models')
    plt.ylabel('Average Latency (ms)')
    plt.title('Model Inference Latency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save:
        file_path = os.path.join(output_dir, 'inference_latency.png')
        plt.savefig(file_path)
        saved_paths.append(file_path)
        logger.info(f"Saved inference latency plot to {file_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return saved_paths
