#!/usr/bin/env python3
"""
Script to generate example datasets.
"""

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.generate_data import save_iris, save_diabetes, save_breast_cancer, save_wine

def main():
    save_iris()
    save_diabetes()
    save_breast_cancer()
    save_wine()

if __name__ == "__main__":
    main()
