"""
Configuration manager for the Distributed Supervised Learning Platform.
"""

import os
import yaml
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return {}


def get_project_root():
    """
    Get the absolute path of the project root directory.
    
    Returns:
        Path: Project root directory path.
    """
    return Path(__file__).parent.parent.parent
