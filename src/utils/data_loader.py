###### Funciones de utilidad para cargar y preprocesar datos

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List

from src.utils.config import get_project_root


def load_dataset(dataset_name: str, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ###### Cargar un dataset desde el directorio de datos
    #
    # Args:
    #    dataset_name (str): Nombre del dataset a cargar
    #    **kwargs: Argumentos adicionales para la carga del dataset
    #
    # Returns:
    #    Tuple[pd.DataFrame, pd.DataFrame]: Dataframes de entrenamiento y prueba
    data_dir = get_project_root() / 'data'
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    # Check if processed data already exists
    train_path = processed_dir / f"{dataset_name}_train.csv"
    test_path = processed_dir / f"{dataset_name}_test.csv"
    
    if train_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"Loaded processed dataset: {dataset_name}")
        return train_df, test_df
    
    # If not, load from raw data and preprocess
    raw_path = raw_dir / f"{dataset_name}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found in {raw_dir}")
    
    # Load raw data
    df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset: {dataset_name} with shape {df.shape}")
    
    # Preprocess data
    train_df, test_df = preprocess_data(df, **kwargs)
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save processed data
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return train_df, test_df


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42,
                    target_column: str = None, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess a dataframe and split it into training and testing sets.
    
    Args:
        df (pd.DataFrame): The dataframe to preprocess.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
        target_column (str): Name of the target column.
        **kwargs: Additional preprocessing parameters.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing dataframes.
    """
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical features
    df = encode_categorical_features(df, target_column)
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    if target_column:
        train_df, test_df = train_test_split(df, test_size=test_size, 
                                             random_state=random_state, 
                                             stratify=df[target_column] if target_column else None)
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, 
                                             random_state=random_state)
    
    return train_df, test_df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with potentially missing values.
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled.
    """
    # For numeric columns, fill with mean
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    # For categorical columns, fill with mode
    cat_cols = df.select_dtypes(exclude=['number']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df


def encode_categorical_features(df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
    """
    Encode categorical features in the dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with categorical features.
        target_column (str): Name of the target column.
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical features.
    """
    # Get categorical columns (excluding the target)
    cat_cols = df.select_dtypes(include=['object']).columns
    if target_column and target_column in cat_cols:
        cat_cols = [col for col in cat_cols if col != target_column]
    
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df_encoded
