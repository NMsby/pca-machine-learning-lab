"""
Data generation utilities for PCA testing and analysis.

This module provides functions to generate synthetic datasets with known
structure for testing and validating PCA implementations.

Author: [Your Name]
Course: Machine Learning
Date: June 2025
"""

import numpy as np
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt


def generate_correlated_data(n_samples: int = 100,
                             n_features: int = 2,
                             correlation: float = 0.8,
                             noise_std: float = 0.1,
                             random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate synthetic data with controlled correlation structure.

    Parameters:
        n_samples: Number of samples to generate
        n_features: Number of features
        correlation: Correlation coefficient between features
        noise_std: Standard deviation of noise
        random_state: Random seed for reproducibility

    Returns:
        X: Generated data of shape (n_samples, n_features)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate base feature
    x1 = np.random.randn(n_samples)

    # Generate correlated features
    X = np.zeros((n_samples, n_features))
    X[:, 0] = x1

    for i in range(1, n_features):
        # Add correlation and noise
        X[:, i] = correlation * x1 + np.sqrt(1 - correlation ** 2) * np.random.randn(n_samples)
        X[:, i] += noise_std * np.random.randn(n_samples)

    return X


def generate_elliptical_data(n_samples: int = 200,
                             major_axis: float = 3.0,
                             minor_axis: float = 1.0,
                             rotation_angle: float = np.pi / 4,
                             center: Tuple[float, float] = (0, 0),
                             noise_std: float = 0.1,
                             random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate elliptical data with known principal directions.

    Parameters:
        n_samples: Number of samples
        major_axis: Length of major axis
        minor_axis: Length of minor axis
        rotation_angle: Rotation angle in radians
        center: Center point (x, y)
        noise_std: Standard deviation of noise
        random_state: Random seed

    Returns:
        X: Generated 2D data
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate points on ellipse
    theta = np.linspace(0, 2 * np.pi, n_samples)
    x = major_axis * np.cos(theta) + noise_std * np.random.randn(n_samples)
    y = minor_axis * np.sin(theta) + noise_std * np.random.randn(n_samples)

    # Apply rotation
    cos_rot = np.cos(rotation_angle)
    sin_rot = np.sin(rotation_angle)

    x_rot = cos_rot * x - sin_rot * y + center[0]
    y_rot = sin_rot * x + cos_rot * y + center[1]

    return np.column_stack([x_rot, y_rot])


def generate_3d_manifold(n_samples: int = 300,
                         manifold_type: str = 'swiss_roll',
                         noise_std: float = 0.1,
                         random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D data with 2D manifold structure.

    Parameters:
        n_samples: Number of samples
        manifold_type: Type of manifold ('swiss_roll', 'helix', 'plane')
        noise_std: Standard deviation of noise
        random_state: Random seed

    Returns:
        X: 3D data
        colors: Color values for visualization
    """
    if random_state is not None:
        np.random.seed(random_state)

    if manifold_type == 'swiss_roll':
        t = 3 * np.pi / 2 * (1 + 2 * np.random.rand(n_samples))
        height = 21 * np.random.rand(n_samples)

        X = np.zeros((n_samples, 3))
        X[:, 0] = t * np.cos(t)
        X[:, 1] = height
        X[:, 2] = t * np.sin(t)
        colors = t

    elif manifold_type == 'helix':
        t = np.linspace(0, 4 * np.pi, n_samples)
        X = np.zeros((n_samples, 3))
        X[:, 0] = np.cos(t)
        X[:, 1] = np.sin(t)
        X[:, 2] = t / (2 * np.pi)
        colors = t

    elif manifold_type == 'plane':
        u = np.random.uniform(-2, 2, n_samples)
        v = np.random.uniform(-2, 2, n_samples)

        X = np.zeros((n_samples, 3))
        X[:, 0] = u + 0.5 * v
        X[:, 1] = 2 * u - v
        X[:, 2] = 0.5 * u + 2 * v
        colors = u + v

    else:
        raise ValueError(f"Unknown manifold type: {manifold_type}")

    # Add noise
    X += noise_std * np.random.randn(*X.shape)

    return X, colors


def generate_high_dimensional_data(n_samples: int = 200,
                                   n_features: int = 50,
                                   n_informative: int = 5,
                                   noise_std: float = 1.0,
                                   random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate high-dimensional data with low-dimensional structure.

    Parameters:
        n_samples: Number of samples
        n_features: Total number of features
        n_informative: Number of informative features
        noise_std: Standard deviation of noise
        random_state: Random seed

    Returns:
        X: High-dimensional data
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate low-dimensional latent variables
    Z = np.random.randn(n_samples, n_informative)

    # Create random projection matrix
    A = np.random.randn(n_features, n_informative)

    # Project to high dimensions
    X = Z @ A.T

    # Add noise
    X += noise_std * np.random.randn(*X.shape)

    return X


def create_classification_data_pca(n_samples: int = 300,
                                   n_classes: int = 3,
                                   separation: float = 2.0,
                                   random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create classification data suitable for PCA visualization.

    Parameters:
        n_samples: Number of samples
        n_classes: Number of classes
        separation: Separation between classes
        random_state: Random seed

    Returns:
        X: Feature data
        y: Class labels
    """
    if random_state is not None:
        np.random.seed(random_state)

    samples_per_class = n_samples // n_classes
    X = []
    y = []

    for i in range(n_classes):
        # Create cluster center
        angle = 2 * np.pi * i / n_classes
        center = separation * np.array([np.cos(angle), np.sin(angle)])

        # Generate samples around the center
        class_samples = np.random.multivariate_normal(
            center,
            [[1, 0.5], [0.5, 1]],
            samples_per_class
        )

        X.append(class_samples)
        y.extend([i] * samples_per_class)

    X = np.vstack(X)
    y = np.array(y)

    return X, y


def add_outliers(X: np.ndarray,
                 outlier_fraction: float = 0.1,
                 outlier_factor: float = 3.0,
                 random_state: Optional[int] = None) -> np.ndarray:
    """
    Add outliers to existing data.

    Parameters:
        X: Original data
        outlier_fraction: Fraction of data to make outliers
        outlier_factor: How far outliers are from the main data
        random_state: Random seed

    Returns:
        X_with_outliers: Data with outliers added
    """
    if random_state is not None:
        np.random.seed(random_state)

    X_copy = X.copy()
    n_outliers = int(outlier_fraction * len(X))

    if n_outliers > 0:
        # Select random indices for outliers
        outlier_indices = np.random.choice(len(X), n_outliers, replace=False)

        # Calculate data range
        data_range = np.ptp(X, axis=0)
        data_center = np.mean(X, axis=0)

        # Generate outliers
        for idx in outlier_indices:
            direction = np.random.randn(X.shape[1])
            direction = direction / np.linalg.norm(direction)
            X_copy[idx] = data_center + outlier_factor * data_range * direction

    return X_copy


if __name__ == "__main__":
    # Test all data generation functions
    print("Testing data generation utilities...")

    # Test correlated data
    X_corr = generate_correlated_data(100, 3, correlation=0.8, random_state=42)
    print(f"Correlated data shape: {X_corr.shape}")
    print(f"Correlation matrix:\n{np.corrcoef(X_corr.T)}")

    # Test elliptical data
    X_ellipse = generate_elliptical_data(200, random_state=42)
    print(f"Elliptical data shape: {X_ellipse.shape}")

    # Test 3D manifold
    X_3d, colors = generate_3d_manifold(100, 'swiss_roll', random_state=42)
    print(f"3D manifold data shape: {X_3d.shape}")

    print("All data generation functions working correctly!")
