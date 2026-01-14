"""
Feature Engineering Module for House Price Prediction

This module contains functions to create engineered features from the
original 7-feature dataset, including interaction terms, ratios, and
composite features.

Author: Abdullah Al Galib
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import Union


def engineer_features(X: Union[np.ndarray, pd.DataFrame],
                      feature_names: list = None) -> np.ndarray:
    """
    Create engineered features from base features.

    Base Features (in order):
    0: OverallQual, 1: GrLivArea, 2: GarageCars, 3: TotalBsmtSF,
    4: 1stFlrSF, 5: FullBath, 6: YearBuilt

    Engineered Features Created:
    - Quality_Size: OverallQual × GrLivArea (luxury premium)
    - Age_Quality: House_Age × OverallQual (depreciation)
    - Bath_Density: FullBath per 1000 sqft
    - Garage_Ratio: GarageCars / GrLivArea
    - Basement_Ratio: TotalBsmtSF / 1stFlrSF
    - Total_Space: GrLivArea + TotalBsmtSF
    - House_Age: 2026 - YearBuilt
    - Is_New: Binary flag (1 if built after 2000)

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame of shape (n_samples, 7)
        Base feature matrix

    feature_names : list, optional
        Names of base features (for validation)

    Returns
    -------
    X_engineered : np.ndarray of shape (n_samples, 15)
        Original 7 features + 8 engineered features
    """
    # Convert to numpy if DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.values

    # Validate shape
    if X.shape[1] != 7:
        raise ValueError(f"Expected 7 base features, got {X.shape[1]}")

    # Extract base features
    OverallQual = X[:, 0]
    GrLivArea = X[:, 1]
    GarageCars = X[:, 2]
    TotalBsmtSF = X[:, 3]
    FirstFlrSF = X[:, 4]
    FullBath = X[:, 5]
    YearBuilt = X[:, 6]

    # Calculate House Age
    House_Age = 2026 - YearBuilt

    # Create engineered features
    Quality_Size = OverallQual * GrLivArea
    Age_Quality = House_Age * OverallQual
    Bath_Density = FullBath / (GrLivArea / 1000 + 1e-6)  # Avoid division by 0
    Garage_Ratio = GarageCars / (GrLivArea + 1e-6)
    Basement_Ratio = TotalBsmtSF / (FirstFlrSF + 1)
    Total_Space = GrLivArea + TotalBsmtSF
    Is_New = (YearBuilt > 2000).astype(float)

    # Stack all features
    X_engineered = np.column_stack([
        # Original 7
        OverallQual, GrLivArea, GarageCars, TotalBsmtSF,
        FirstFlrSF, FullBath, YearBuilt,
        # Engineered 8
        Quality_Size, Age_Quality, Bath_Density, Garage_Ratio,
        Basement_Ratio, Total_Space, House_Age, Is_New
    ])

    return X_engineered


def get_feature_names() -> list:
    """
    Get names of all features (base + engineered).

    Returns
    -------
    names : list of str
        List of 15 feature names
    """
    return [
        # Base 7
        'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
        '1stFlrSF', 'FullBath', 'YearBuilt',
        # Engineered 8
        'Quality_Size', 'Age_Quality', 'Bath_Density', 'Garage_Ratio',
        'Basement_Ratio', 'Total_Space', 'House_Age', 'Is_New'
    ]


def engineer_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features directly on a DataFrame with named columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: OverallQual, GrLivArea, GarageCars, TotalBsmtSF,
        1stFlrSF, FullBath, YearBuilt

    Returns
    -------
    df_engineered : pd.DataFrame
        DataFrame with original + engineered features
    """
    df = df.copy()

    # Calculate House Age first
    df['House_Age'] = 2026 - df['YearBuilt']

    # Interaction features
    df['Quality_Size'] = df['OverallQual'] * df['GrLivArea']
    df['Age_Quality'] = df['House_Age'] * df['OverallQual']

    # Ratio features
    df['Bath_Density'] = df['FullBath'] / (df['GrLivArea'] / 1000 + 1e-6)
    df['Garage_Ratio'] = df['GarageCars'] / (df['GrLivArea'] + 1e-6)
    df['Basement_Ratio'] = df['TotalBsmtSF'] / (df['1stFlrSF'] + 1)

    # Composite features
    df['Total_Space'] = df['GrLivArea'] + df['TotalBsmtSF']
    df['Is_New'] = (df['YearBuilt'] > 2000).astype(int)

    return df


if __name__ == "__main__":
    # Test the function
    print("Testing feature engineering...")

    # Create dummy data
    X_test = np.array([
        [7, 1500, 2, 1000, 1000, 2, 2000],  # Average house
        [10, 3000, 3, 2000, 2000, 3, 2020],  # Luxury new house
        [3, 800, 1, 500, 500, 1, 1950]  # Old small house
    ])

    X_eng = engineer_features(X_test)
    feature_names = get_feature_names()

    print(f"\nOriginal shape: {X_test.shape}")
    print(f"Engineered shape: {X_eng.shape}")
    print(f"\nFeature names: {feature_names}")
    print(f"\nSample engineered features:")
    print(f"Quality_Size (house 1): {X_eng[0, 7]:.0f}")
    print(f"House_Age (house 3): {X_eng[2, 13]:.0f}")
    print("\n✓ Feature engineering working correctly!")
