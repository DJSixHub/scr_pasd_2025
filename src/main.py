#!/usr/bin/env python3
"""
Main entry point for the Distributed Supervised Learning Platform.
"""

import argparse
import os
import sys
import ray
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.training.trainer import DistributedTrainer
from src.serving.api import ModelServing
from src.monitoring.monitor import SystemMonitor


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Supervised Learning Platform")
    parser.add_argument("--mode", type=str, choices=['train', 'serve', 'monitor'], 
                        required=True, help="Operation mode")
    parser.add_argument("--config", type=str, default="src/config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--head", action="store_true", 
                        help="Start as head node (for train mode)")
    parser.add_argument("--worker", action="store_true", 
                        help="Start as worker node (for train mode)")
    parser.add_argument("--head-address", type=str, default=None,
                        help="Address of the head node (for worker nodes)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize Ray based on node type
    if args.mode == 'train':
        if args.head:
            ray.init(include_dashboard=True)
            print("Ray initialized as head node")
        elif args.worker and args.head_address:
            ray.init(address=args.head_address)
            print(f"Ray initialized as worker node, connected to {args.head_address}")
        else:
            print("Error: For training mode, specify either --head or --worker with --head-address")
            return
        
        # Start training
        trainer = DistributedTrainer(config)
        trainer.train()
        
    elif args.mode == 'serve':
        # Initialize Ray for serving
        ray.init()
        serving = ModelServing(config)
        serving.start()
        
    elif args.mode == 'monitor':
        # Initialize Ray for monitoring
        ray.init()
        monitor = SystemMonitor(config)
        monitor.start()
        
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
