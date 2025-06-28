"""
Simplified data loading utilities for the Streamlit-driven ML platform
"""
import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def load_dataset(file_path, header='infer', index_col=None):
    """
    Load a dataset from a file.
    
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

def preprocess_dataset(df, target_col=None, test_size=0.2, random_state=42, scale=False):
    """
    Preprocess a dataset.
    
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
