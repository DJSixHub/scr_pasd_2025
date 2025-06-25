import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_diabetes, 
    fetch_california_housing, load_digits, make_classification,
    make_regression
)
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load real datasets from scikit-learn
real_datasets = {
    'iris': load_iris(),
    'wine': load_wine(),
    'breast_cancer': load_breast_cancer(),
    'diabetes': load_diabetes(),
    'digits': load_digits(),
}

# California housing dataset (larger dataset)
try:
    california = fetch_california_housing()
    real_datasets['california_housing'] = california
except Exception as e:
    print(f"Could not fetch California housing dataset: {e}")

# Save real datasets as CSV
for name, dataset in real_datasets.items():
    df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    
    output_path = os.path.join('data', f'{name}.csv')
    df.to_csv(output_path, index=False)

# Generate synthetic classification datasets with different characteristics
synthetic_classification = {
    'binary_classification': make_classification(n_samples=1000, n_features=20, n_informative=10, 
                                        n_redundant=5, n_classes=2, random_state=42),
    'multi_classification': make_classification(n_samples=1000, n_features=20, n_informative=10, 
                                      n_redundant=5, n_classes=3, random_state=43),
    'imbalanced_classification': make_classification(n_samples=1000, n_features=20, n_informative=10, 
                                          weights=[0.9, 0.1], n_classes=2, random_state=44)
}

# Save synthetic classification datasets
for name, (X, y) in synthetic_classification.items():
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(data=X, columns=feature_names)
    df['target'] = y
    
    output_path = os.path.join('data', f'{name}.csv')
    df.to_csv(output_path, index=False)

# Generate synthetic regression datasets
synthetic_regression = {
    'simple_regression': make_regression(n_samples=1000, n_features=10, noise=0.5, random_state=42),
    'complex_regression': make_regression(n_samples=1000, n_features=20, n_informative=10, noise=0.5, random_state=43)
}

# Save synthetic regression datasets
for name, (X, y) in synthetic_regression.items():
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(data=X, columns=feature_names)
    df['target'] = y
    
    output_path = os.path.join('data', f'{name}.csv')
    df.to_csv(output_path, index=False)

print(f"Created {len(real_datasets) + len(synthetic_classification) + len(synthetic_regression)} datasets in ./data/")
