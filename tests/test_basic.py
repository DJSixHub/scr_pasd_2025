"""
Test script for the Distributed Supervised Learning Platform.
"""

import os
import time
import ray
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.data_loader import load_dataset
from src.models.model_factory import get_model


def test_data_loader():
    """Test the data loader functionality."""
    print("\n=== Testing Data Loader ===")
    try:
        train_df, test_df = load_dataset('iris', target_column='species')
        print(f"Loaded iris dataset: train shape={train_df.shape}, test shape={test_df.shape}")
        print(f"Train columns: {train_df.columns.tolist()}")
        print(f"Sample data:\n{train_df.head(3)}")
        return True
    except Exception as e:
        print(f"Error testing data loader: {e}")
        return False


def test_model_factory():
    """Test the model factory functionality."""
    print("\n=== Testing Model Factory ===")
    model_types = ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'knn', 'decision_tree']
    success = True
    
    for model_type in model_types:
        try:
            model = get_model(model_type, task='classification')
            print(f"Successfully created {model_type} model: {type(model)}")
        except Exception as e:
            print(f"Error creating {model_type} model: {e}")
            success = False
    
    return success


def test_training_workflow():
    """Test the basic training workflow."""
    print("\n=== Testing Basic Training Workflow ===")
    try:
        # Load dataset
        train_df, test_df = load_dataset('iris', target_column='species')
        
        # Prepare data
        X_train = train_df.drop(columns=['species'])
        y_train = train_df['species']
        X_test = test_df.drop(columns=['species'])
        y_test = test_df['species']
        
        # Create and train model
        model = get_model('random_forest', task='classification', n_estimators=100)
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        accuracy = model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.4f}")
        
        return True
    except Exception as e:
        print(f"Error testing training workflow: {e}")
        return False


def test_ray_initialization():
    """Test Ray initialization."""
    print("\n=== Testing Ray Initialization ===")
    try:
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        print("Ray initialized successfully")
        print(f"Ray Dashboard URL: {ray.get_dashboard_url()}")
        
        # Get cluster info
        cluster_info = ray.cluster_resources()
        print(f"Cluster resources: {cluster_info}")
        
        # Shutdown Ray
        ray.shutdown()
        print("Ray shutdown successfully")
        
        return True
    except Exception as e:
        print(f"Error testing Ray initialization: {e}")
        try:
            ray.shutdown()
        except:
            pass
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("Starting tests for Distributed Supervised Learning Platform...")
    
    # Ensure data is generated
    print("Checking for example datasets...")
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / 'data' / 'raw'
    if not (raw_data_dir / 'iris.csv').exists():
        print("Example datasets not found. Generating...")
        from src.utils.generate_data import save_iris, save_diabetes, save_breast_cancer, save_wine
        save_iris()
        save_diabetes()
        save_breast_cancer()
        save_wine()
    
    # Run tests
    results = {
        "Data Loader": test_data_loader(),
        "Model Factory": test_model_factory(),
        "Training Workflow": test_training_workflow(),
        "Ray Initialization": test_ray_initialization()
    }
    
    # Report results
    print("\n=== Test Results ===")
    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the output above for details.")


if __name__ == "__main__":
    run_all_tests()
