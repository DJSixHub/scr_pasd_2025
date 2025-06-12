#!/usr/bin/env python3
"""
Distributed Supervised Learning Platform - Demo Script

This script demonstrates how to use the platform to train models on multiple datasets,
deploy them for serving, and monitor the system.
"""

import os
import sys
import time
import argparse
import subprocess
import threading
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Supervised Learning Platform Demo")
    parser.add_argument("--demo-mode", type=str, choices=['all', 'train', 'serve', 'monitor'], 
                        default='all', help="Demo mode")
    parser.add_argument("--config", type=str, default="src/config/config.yaml",
                        help="Path to configuration file")
    return parser.parse_args()


def run_generate_data():
    """Generate example datasets."""
    print("\n=== Generating Example Datasets ===")
    try:
        from src.utils.generate_data import save_iris, save_diabetes, save_breast_cancer, save_wine
        save_iris()
        save_diabetes()
        save_breast_cancer()
        save_wine()
        print("Example datasets generated successfully!")
        return True
    except Exception as e:
        print(f"Error generating datasets: {e}")
        return False


def run_head_node(config_path):
    """Run the head node for training."""
    print("\n=== Starting Head Node ===")
    try:
        cmd = ["python", "src/main.py", "--mode=train", "--head", f"--config={config_path}"]
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error starting head node: {e}")
        return None


def run_worker_node(config_path, head_address="localhost:6379"):
    """Run a worker node for training."""
    print(f"\n=== Starting Worker Node (connecting to {head_address}) ===")
    try:
        cmd = ["python", "src/main.py", "--mode=train", "--worker", 
               f"--head-address={head_address}", f"--config={config_path}"]
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error starting worker node: {e}")
        return None


def run_serving(config_path):
    """Run the model serving API."""
    print("\n=== Starting Model Serving API ===")
    try:
        cmd = ["python", "src/main.py", "--mode=serve", f"--config={config_path}"]
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error starting model serving: {e}")
        return None


def run_monitoring(config_path):
    """Run the system monitoring."""
    print("\n=== Starting System Monitoring ===")
    try:
        cmd = ["python", "src/main.py", "--mode=monitor", f"--config={config_path}"]
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error starting monitoring: {e}")
        return None


def main():
    args = parse_args()
      # Check if configuration file exists, if not create a default one
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} not found. Creating default configuration...")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Create a basic configuration file
        with open(config_path, 'w') as f:
            f.write("""###### Configuration file for Distributed Supervised Learning Platform

# Training Configuration
datasets:
  - name: iris
    target_column: species
    test_size: 0.2
    random_state: 42
  
  - name: diabetes
    target_column: target
    test_size: 0.25
    random_state: 42

# Models Configuration
models:
  - type: random_forest
    params:
      n_estimators: 100
      max_depth: 10
      random_state: 42
      task: classification
  
  - type: gradient_boosting
    params:
      n_estimators: 100
      learning_rate: 0.1
      random_state: 42
      task: classification
  
  - type: logistic_regression
    params:
      max_iter: 1000
      random_state: 42
  
  - type: svm
    params:
      kernel: rbf
      C: 1.0
      random_state: 42
      task: classification

# Ray Configuration
ray:
  head_address: localhost:6379
  redis_password: null
  num_cpus: null  # Use all available CPUs
  num_gpus: 0

# Service Configuration
serving:
  port: 8000
  host: 0.0.0.0

# Monitoring Configuration
monitoring:
  interval: 5  # in seconds
  save_plots: true
""")
        print(f"Created default configuration at {config_path}")
    
    # Generate example datasets if they don't exist
    data_dir = Path("data/raw")
    if not (data_dir / "iris.csv").exists():
        if not run_generate_data():
            print("Failed to generate example datasets. Exiting.")
            return
    
    processes = []
    
    if args.demo_mode in ['all', 'train']:
        # Start the head node
        head_process = run_head_node(config_path)
        if head_process:
            processes.append(head_process)
            
            # Give the head node time to initialize
            print("Waiting for head node to initialize...")
            time.sleep(10)
            
            # Start worker nodes
            for i in range(2):  # Start 2 worker nodes
                worker_process = run_worker_node(config_path)
                if worker_process:
                    processes.append(worker_process)
                time.sleep(2)  # Give workers time to connect
    
    if args.demo_mode in ['all', 'serve']:
        # Start model serving
        serving_process = run_serving(config_path)
        if serving_process:
            processes.append(serving_process)
    
    if args.demo_mode in ['all', 'monitor']:
        # Start monitoring
        monitor_process = run_monitoring(config_path)
        if monitor_process:
            processes.append(monitor_process)
    
    print("\n=== Demo Running ===")
    print("Press Ctrl+C to stop the demo...")
    
    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("\nStopping all processes...")
        for process in processes:
            process.terminate()
        
        # Wait for processes to terminate
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        print("All processes stopped.")


if __name__ == "__main__":
    main()
