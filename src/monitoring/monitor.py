###### Este módulo proporciona funcionalidad para monitorear el sistema distribuido

import time
import ray
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path

from src.utils.config import get_project_root


@ray.remote
class MetricsCollector:
    ###### Actor de Ray para recolectar métricas del sistema
    
    def __init__(self):
        """
        Initialize the metrics collector.
        """
        self.node_metrics = {}
        self.start_time = time.time()
    
    def collect_system_metrics(self, node_id):
        """
        Collect system metrics for the current node.
        
        Args:
            node_id (str): ID of the node.
            
        Returns:
            Dict: System metrics.
        """
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "timestamp": time.time() - self.start_time
        }
        
        # Store metrics for this node
        if node_id not in self.node_metrics:
            self.node_metrics[node_id] = []
        
        self.node_metrics[node_id].append(metrics)
        
        return metrics
    
    def get_all_metrics(self):
        """
        Get all collected metrics.
        
        Returns:
            Dict: All metrics.
        """
        return self.node_metrics


class SystemMonitor:
    """
    Main class for system monitoring.
    """
    
    def __init__(self, config):
        """
        Initialize the system monitor.
        
        Args:
            config (Dict): Configuration for the monitor.
        """
        self.config = config
        self.plots_dir = get_project_root() / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        self.metrics_collector = None
        self.nodes = []
    
    def start(self):
        """
        Start the system monitoring.
        """
        print("Starting system monitoring...")
        
        # Create metrics collector actor
        self.metrics_collector = MetricsCollector.remote()
        
        # Monitor cluster nodes
        try:
            # Get all nodes in the Ray cluster
            nodes = ray.nodes()
            self.nodes = [node["NodeID"] for node in nodes]
            
            print(f"Monitoring {len(self.nodes)} nodes in the cluster")
            
            # Start monitoring loop
            try:
                self._monitoring_loop()
            except KeyboardInterrupt:
                # Generate final report on exit
                self._generate_report()
                print("Monitoring stopped")
        
        except Exception as e:
            print(f"Error monitoring cluster: {e}")
    
    def _monitoring_loop(self):
        """
        Main monitoring loop.
        """
        interval = self.config.get("monitoring", {}).get("interval", 5)
        
        while True:
            # Collect metrics from all nodes
            for node_id in self.nodes:
                ray.get(self.metrics_collector.collect_system_metrics.remote(node_id))
            
            # Sleep for the specified interval
            time.sleep(interval)
    
    def _generate_report(self):
        """
        Generate monitoring reports and plots.
        """
        print("Generating monitoring reports...")
        
        # Get all metrics
        metrics = ray.get(self.metrics_collector.get_all_metrics.remote())
        
        # Generate plots
        self._plot_cpu_usage(metrics)
        self._plot_memory_usage(metrics)
        
        print(f"Reports generated in {self.plots_dir}")
    
    def _plot_cpu_usage(self, metrics):
        """
        Plot CPU usage over time.
        
        Args:
            metrics (Dict): Metrics to plot.
        """
        plt.figure(figsize=(12, 6))
        
        for node_id, node_metrics in metrics.items():
            if not node_metrics:
                continue
            
            df = pd.DataFrame(node_metrics)
            plt.plot(df["timestamp"], df["cpu_percent"], label=f"Node {node_id[:8]}")
        
        plt.title("CPU Usage Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("CPU Usage (%)")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.plots_dir / "cpu_usage.png")
        plt.close()
    
    def _plot_memory_usage(self, metrics):
        """
        Plot memory usage over time.
        
        Args:
            metrics (Dict): Metrics to plot.
        """
        plt.figure(figsize=(12, 6))
        
        for node_id, node_metrics in metrics.items():
            if not node_metrics:
                continue
            
            df = pd.DataFrame(node_metrics)
            plt.plot(df["timestamp"], df["memory_percent"], label=f"Node {node_id[:8]}")
        
        plt.title("Memory Usage Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory Usage (%)")
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.plots_dir / "memory_usage.png")
        plt.close()
