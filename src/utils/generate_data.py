"""
Generate example datasets for the Distributed Supervised Learning Platform.
"""

import os
import numpy as np
import pandas as pd
from sklearn import datasets
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent.parent
data_dir = project_root / 'data'
raw_dir = data_dir / 'raw'

# Create directories if they don't exist
os.makedirs(raw_dir, exist_ok=True)

def save_iris():
    """Save the Iris dataset to CSV."""
    print("Generating Iris dataset...")
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=iris['feature_names'] + ['species'])
    # Convert numeric target to string class for better readability
    target_names = iris.target_names
    iris_df['species'] = iris_df['species'].astype(int).map(
        {i: name for i, name in enumerate(target_names)}
    )
    
    # Save to CSV
    iris_path = raw_dir / 'iris.csv'
    iris_df.to_csv(iris_path, index=False)
    print(f"Iris dataset saved to {iris_path}")

def save_diabetes():
    """Save the Diabetes dataset to CSV."""
    print("Generating Diabetes dataset...")
    diabetes = datasets.load_diabetes()
    diabetes_df = pd.DataFrame(data=diabetes['data'],
                              columns=diabetes['feature_names'])
    diabetes_df['target'] = diabetes['target']
    
    # Save to CSV
    diabetes_path = raw_dir / 'diabetes.csv'
    diabetes_df.to_csv(diabetes_path, index=False)
    print(f"Diabetes dataset saved to {diabetes_path}")

def save_breast_cancer():
    """Save the Breast Cancer dataset to CSV."""
    print("Generating Breast Cancer dataset...")
    cancer = datasets.load_breast_cancer()
    cancer_df = pd.DataFrame(data=cancer['data'],
                            columns=cancer['feature_names'])
    cancer_df['target'] = cancer['target']
    
    # Save to CSV
    cancer_path = raw_dir / 'breast_cancer.csv'
    cancer_df.to_csv(cancer_path, index=False)
    print(f"Breast Cancer dataset saved to {cancer_path}")

def save_wine():
    """Save the Wine dataset to CSV."""
    print("Generating Wine dataset...")
    wine = datasets.load_wine()
    wine_df = pd.DataFrame(data=np.c_[wine['data'], wine['target']],
                          columns=wine['feature_names'] + ['target'])
    
    # Save to CSV
    wine_path = raw_dir / 'wine.csv'
    wine_df.to_csv(wine_path, index=False)
    print(f"Wine dataset saved to {wine_path}")

if __name__ == "__main__":
    print("Generating example datasets...")
    save_iris()
    save_diabetes()
    save_breast_cancer()
    save_wine()
    print("All datasets generated successfully!")
