"""
Visualization utilities for monitoring models and system performance
"""
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import io
import base64
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)

def plot_training_metrics(metrics_dict, output_dir='plots', save=True, show=True):
    """
    Plot training metrics for multiple models
    
    Args:
        metrics_dict (dict): Dictionary mapping model names to metrics dictionaries or scalar metrics
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
    
    # Handle case where metrics_dict values are scalar (not dictionaries)
    # Convert simple scalar metrics to a dictionary with a single 'performance' key
    is_scalar_metrics = False
    first_value = next(iter(metrics_dict.values())) if metrics_dict else None
    if first_value is not None and not isinstance(first_value, dict):
        is_scalar_metrics = True
        # Convert to a uniform format
        metrics_dict_normalized = {
            model_name: {'performance': metric_value}
            for model_name, metric_value in metrics_dict.items()
        }
    else:
        metrics_dict_normalized = metrics_dict
    
    # Determine which metrics are available
    all_metric_keys = set()
    for model_metrics in metrics_dict_normalized.values():
        all_metric_keys.update(model_metrics.keys())
    
    # Remove training_time for separate plot
    metric_keys = [key for key in all_metric_keys if key != 'training_time']
    
    # Plot performance metrics
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics_dict_normalized))
    width = 0.8 / max(len(metric_keys), 1)  # Avoid division by zero
    
    for i, metric_name in enumerate(metric_keys):
        values = [metrics.get(metric_name, 0) for model, metrics in metrics_dict_normalized.items()]
        plt.bar(x + i * width, values, width, label=metric_name)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    
    # Set appropriate title based on metrics type
    if is_scalar_metrics:
        performance_type = "Accuracy" if max(metrics_dict.values(), default=0) <= 1.0 else "Performance"
        plt.title(f'Model {performance_type} Comparison')
    else:
        plt.title('Model Performance Metrics')
        
    plt.xticks(x + width * (len(metric_keys) - 1) / 2, metrics_dict_normalized.keys(), rotation=45)
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
    training_times = [metrics.get('training_time', 0) for model, metrics in metrics_dict_normalized.items()]
    
    plt.bar(metrics_dict_normalized.keys(), training_times, color='skyblue')
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

def plot_roc_curve_to_png(roc_data, model_name, return_base64=False):
    """
    Generate ROC curve plot as PNG image
    
    Args:
        roc_data (dict): ROC curve data with fpr, tpr, auc for each class
        model_name (str): Name of the model
        return_base64 (bool): Whether to return base64 encoded string
        
    Returns:
        bytes or str: PNG image bytes or base64 encoded string
    """
    plt.figure(figsize=(10, 8))
    
    if roc_data.get('type') == 'multiclass':
        # Plot ROC curve for each class
        classes = roc_data.get('classes', {})
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (class_name, class_data) in enumerate(classes.items()):
            if 'fpr' in class_data and 'tpr' in class_data:
                color = colors[i % len(colors)]
                plt.plot(
                    class_data['fpr'], 
                    class_data['tpr'], 
                    color=color,
                    lw=2, 
                    label=f'{class_name} (AUC = {class_data.get("auc", 0.0):.2f})'
                )
    else:
        # Binary classification
        if 'fpr' in roc_data and 'tpr' in roc_data:
            plt.plot(
                roc_data['fpr'], 
                roc_data['tpr'], 
                color='darkorange',
                lw=2, 
                label=f'ROC curve (AUC = {roc_data.get("auc", 0.0):.2f})'
            )
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save to bytes
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    
    if return_base64:
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return img_base64
    else:
        img_bytes = buffer.getvalue()
        plt.close()
        return img_bytes

def plot_learning_curve_to_png(learning_data, model_name, return_base64=False):
    """
    Generate learning curve plot as PNG image
    
    Args:
        learning_data (dict): Learning curve data with train_sizes, scores, etc.
        model_name (str): Name of the model
        return_base64 (bool): Whether to return base64 encoded string
        
    Returns:
        bytes or str: PNG image bytes or base64 encoded string
    """
    plt.figure(figsize=(10, 6))
    
    train_sizes = learning_data.get('train_sizes', [])
    train_scores_mean = learning_data.get('train_scores_mean', [])
    train_scores_std = learning_data.get('train_scores_std', [])
    val_scores_mean = learning_data.get('val_scores_mean', [])
    val_scores_std = learning_data.get('val_scores_std', [])
    
    if train_sizes and train_scores_mean and val_scores_mean:
        # Plot training scores
        plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
        if train_scores_std:
            plt.fill_between(train_sizes, 
                           np.array(train_scores_mean) - np.array(train_scores_std),
                           np.array(train_scores_mean) + np.array(train_scores_std), 
                           alpha=0.2, color='blue')
        
        # Plot validation scores
        plt.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Validation score')
        if val_scores_std:
            plt.fill_between(train_sizes, 
                           np.array(val_scores_mean) - np.array(val_scores_std),
                           np.array(val_scores_mean) + np.array(val_scores_std), 
                           alpha=0.2, color='red')
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Learning Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save to bytes
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    
    if return_base64:
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return img_base64
    else:
        img_bytes = buffer.getvalue()
        plt.close()
        return img_bytes
