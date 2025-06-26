"""
Data loading and preprocessing utilities with distributed redundancy support
"""
import os
import pandas as pd
import numpy as np
import ray
import logging
import glob
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@ray.remote(max_retries=3)  # Add retries for fault tolerance
def load_dataset(file_path, header='infer', index_col=None):
    """
    Load a dataset from a file in a distributed manner.
    
    Args:
        file_path (str): Path to the dataset file
        header (str or int): Header row. Default is 'infer'
        index_col (int or str, optional): Column to use as index. Default is None
        
    Returns:
        pandas.DataFrame or None: The loaded dataset or None if loading failed
    """
    try:
        logger.info(f"Loading dataset from {file_path}")
        
        # Check file extension to determine loading method
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=header, index_col=index_col)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path, header=header, index_col=index_col)
        else:
            logger.error(f"Unsupported file format: {file_path}")
            return None
            
        logger.info(f"Successfully loaded dataset with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
        return None

@ray.remote(max_retries=3)
def preprocess_dataset(df, target_col=None, test_size=0.2, random_state=42, scale=False):
    """
    Preprocess a dataset in a distributed manner.
    
    Args:
        df (pandas.DataFrame): The dataset to preprocess
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        scale (bool): Whether to scale the features
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) or None if preprocessing fails
    """
    try:
        if df is None or df.empty:
            logger.error("Cannot preprocess empty dataset")
            return None
            
        logger.info(f"Preprocessing dataset with shape {df.shape}")
        
        # Handle missing values
        df = df.dropna()
        
        if target_col is not None:
            if target_col not in df.columns:
                logger.error(f"Target column '{target_col}' not found in dataset")
                return None
                
            # Split features and target
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            
            # Convert categorical features to numeric
            X = pd.get_dummies(X)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features if requested
            if scale:
                scaler = StandardScaler()
                X_train = pd.DataFrame(
                    scaler.fit_transform(X_train), 
                    columns=X_train.columns, 
                    index=X_train.index
                )
                X_test = pd.DataFrame(
                    scaler.transform(X_test), 
                    columns=X_test.columns, 
                    index=X_test.index
                )
                
            logger.info(f"Preprocessing complete: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
            return X_train, X_test, y_train, y_test
        else:
            # If no target column provided, just clean and return the data
            # Convert categorical features to numeric
            df = pd.get_dummies(df)
            
            if scale:
                scaler = StandardScaler()
                df = pd.DataFrame(
                    scaler.fit_transform(df), 
                    columns=df.columns, 
                    index=df.index
                )
                
            logger.info(f"Preprocessing complete: df shape {df.shape}")
            return df, None, None, None
            
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        return None

def get_data_locations(dataset_name, data_dirs=[]):
    """
    Find all locations where a dataset exists for redundancy.
    
    Args:
        dataset_name (str): Base name of the dataset file
        data_dirs (list): List of directories to search for the dataset
        
    Returns:
        list: List of paths where the dataset exists
    """
    if not data_dirs:
        # Default locations to search for data
        data_dirs = [
            os.path.join(os.getcwd(), 'data'),
            '/app/data',
            '/tmp/data',
            os.environ.get('DATA_DIR', ''),
        ]
        
    # Filter out empty paths
    data_dirs = [d for d in data_dirs if d]
    
    all_locations = []
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            # Look for exact matches
            exact_match = os.path.join(data_dir, dataset_name)
            if os.path.exists(exact_match):
                all_locations.append(exact_match)
            
            # Also look for files with the same base name but different extensions
            base_name = os.path.splitext(dataset_name)[0]
            pattern = os.path.join(data_dir, f"{base_name}.*")
            matches = glob.glob(pattern)
            all_locations.extend([m for m in matches if m not in all_locations])
    
    return all_locations

def verify_data_integrity(file_path, expected_hash=None):
    """
    Verify data integrity using file hash.
    
    Args:
        file_path (str): Path to the file
        expected_hash (str): Expected hash value, if available
        
    Returns:
        bool: True if file integrity is verified or no hash is provided
    """
    if not expected_hash:
        return True
        
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash == expected_hash
    except Exception as e:
        logger.error(f"Error verifying data integrity: {e}")
        return False

def load_and_preprocess_data(file_paths, target_col=None, test_size=0.2, random_state=42, scale=False, data_dirs=[], expected_hashes=None):
    """
    Load and preprocess multiple datasets in parallel using Ray with redundancy support.
    
    Args:
        file_paths (list): List of paths to dataset files
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        scale (bool): Whether to scale the features
        data_dirs (list): Additional directories to search for datasets
        expected_hashes (dict): Dictionary mapping file names to expected hashes
        
    Returns:
        list: List of processed datasets
    """
    if not ray.is_initialized():
        logger.warning("Ray is not initialized, initializing now")
        ray.init()
        
    try:
        logger.info(f"Loading and preprocessing {len(file_paths)} datasets with redundancy")
        
        # For each dataset, find all available locations
        dataset_locations = {}
        for file_path in file_paths:
            base_name = os.path.basename(file_path)
            locations = [file_path]  # Include the original path
            
            # Add redundant locations
            redundant_locations = get_data_locations(base_name, data_dirs)
            for loc in redundant_locations:
                if loc not in locations:
                    locations.append(loc)
                    
            dataset_locations[base_name] = locations
            
        # Load datasets in parallel with retry logic for fault tolerance
        dataset_refs = []
        for base_name, locations in dataset_locations.items():
            # Try to load from each location until success
            for location in locations:
                # Verify data integrity if hash is provided
                if expected_hashes and base_name in expected_hashes:
                    if not verify_data_integrity(location, expected_hashes[base_name]):
                        logger.warning(f"Data integrity check failed for {location}, skipping")
                        continue
                        
                dataset_refs.append(load_dataset.remote(location))
                logger.info(f"Attempting to load {base_name} from {location}")
                break  # Stop after first attempt, let Ray's retry mechanism handle failures
                
        # Use ray.get with timeout to handle potential failures
        datasets = []
        for ref in dataset_refs:
            try:
                # Add timeout to prevent indefinite waiting
                dataset = ray.get(ref, timeout=60)
                if dataset is not None:
                    datasets.append(dataset)
            except Exception as e:
                logger.error(f"Failed to get dataset result: {e}")
        
        if not datasets:
            logger.error("All dataset loading attempts failed")
            return []
            
        # Preprocess datasets in parallel
        preprocessing_refs = [
            preprocess_dataset.remote(df, target_col, test_size, random_state, scale)
            for df in datasets if df is not None
        ]
        
        processed_datasets = []
        for ref in preprocessing_refs:
            try:
                # Add timeout to prevent indefinite waiting
                result = ray.get(ref, timeout=120)
                if result is not None:
                    processed_datasets.append(result)
            except Exception as e:
                logger.error(f"Failed to preprocess dataset: {e}")
        
        logger.info(f"Successfully processed {len(processed_datasets)} datasets")
        return processed_datasets
    except Exception as e:
        logger.error(f"Error in load_and_preprocess_data: {e}")
        return []
