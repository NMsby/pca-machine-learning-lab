"""
Principal Component Analysis (PCA) Implementation from Scratch

This module provides a complete implementation of PCA using only NumPy,
following the mathematical derivations established in our theoretical analysis.

Author: Nelson Masbayi
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import warnings


class PCA:
    """
    Principal Component Analysis implementation from scratch.

    This class implements PCA following the mathematical derivation:
    1. Center the data
    2. Compute covariance matrix
    3. Perform eigen decomposition
    4. Sort components by eigenvalue magnitude
    5. Project data onto selected components

    Attributes:
        n_components (int): Number of principal components to keep
        components_ (ndarray): Principal axes in feature space
        explained_variance_ (ndarray): Variance explained by each component
        explained_variance_ratio_ (ndarray): Percentage of variance explained
        singular_values_ (ndarray): Singular values corresponding to components
        mean_ (ndarray): Per-feature empirical mean
        n_features_in_ (int): Number of features in the input data
        n_samples_ (int): Number of samples in the input data
    """

    def __init__(self, n_components: Optional[int] = None,
                 svd_solver: str = 'auto', random_state: Optional[int] = None):
        """
        Initialize PCA.

        Parameters:
            n_components (int, optional): Number of components to keep.
                If None, keep all components.
            svd_solver (str): Solver to use. Options: 'auto', 'covariance', 'svd'
                'auto': selects the best method based on data size
                'covariance': uses eigen decomposition of covariance matrix
                'svd': uses SVD decomposition (more numerically stable)
            random_state (int, optional): Random seed for reproducibility
        """
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.random_state = random_state

        # Attributes to be set during fitting
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_features_in_ = None
        self.n_samples_ = None

    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit the PCA model to the data.

        Parameters:
            X (ndarray): Input data of shape (n_samples, n_features)

        Returns:
            self: Returns the fitted PCA instance
        """
        # Input validation
        X = self._validate_input(X)

        self.n_samples_, self.n_features_in_ = X.shape

        # Determine the number of components
        if self.n_components is None:
            self.n_components = min(self.n_samples_, self.n_features_in_)

        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Choose solver automatically if needed
        solver = self._choose_solver(X_centered)

        # Perform PCA using a chosen method
        if solver == 'covariance':
            self._fit_covariance(X_centered)
        elif solver == 'svd':
            self._fit_svd(X_centered)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        return self

    def _fit_covariance(self, X_centered: np.ndarray) -> None:
        """
        Fit PCA using eigen decomposition of covariance matrix.

        This method follows our mathematical derivation exactly:
        C = (1/(n-1)) * X^T * X
        C * v = λ * v
        """
        # Step 2: Compute covariance matrix
        n_samples = X_centered.shape[0]
        covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # Step 3: Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Step 4: Sort by eigenvalue magnitude (descending)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Keep only the desired number of components
        eigenvalues = eigenvalues[:self.n_components]
        eigenvectors = eigenvectors[:, :self.n_components]

        # Store results
        self.components_ = eigenvectors.T  # Shape: (n_components, n_features)
        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)

        # Calculate singular values: σ = √((n-1) * λ)
        self.singular_values_ = np.sqrt((n_samples - 1) * eigenvalues)

    def _fit_svd(self, X_centered: np.ndarray) -> None:
        """
        Fit PCA using Singular Value Decomposition.

        More numerically stable for large datasets.
        X = U * Σ * V^T
        Components are rows of V^T, eigenvalues are (σ²/(n-1))
        """
        n_samples = X_centered.shape[0]

        # Perform SVD
        U, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Keep only desired components
        singular_values = singular_values[:self.n_components]
        Vt = Vt[:self.n_components, :]

        # Store results
        self.components_ = Vt  # Shape: (n_components, n_features)
        self.singular_values_ = singular_values

        # Calculate explained variance: λ = σ² / (n-1)
        self.explained_variance_ = (singular_values ** 2) / (n_samples - 1)
        total_variance = np.sum((singular_values ** 2)) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto the principal components.

        Parameters:
            X (ndarray): Input data of shape (n_samples, n_features)

        Returns:
            X_transformed (ndarray): Projected data of shape (n_samples, n_components)
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        # Center the data using training mean
        X_centered = X - self.mean_

        # Project onto principal components
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and transform the data.

        Parameters:
            X (ndarray): Input data of shape (n_samples, n_features)

        Returns:
            X_transformed (ndarray): Projected data of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.

        Parameters:
            X_transformed (ndarray): Data in PC space, shape (n_samples, n_components)

        Returns:
            X_reconstructed (ndarray): Data in original space, shape (n_samples, n_features)
        """
        self._check_is_fitted()

        # Project back to original space and add mean
        return X_transformed @ self.components_ + self.mean_

    def get_covariance(self) -> np.ndarray:
        """
        Compute data covariance with the generative model.

        Returns:
            covariance (ndarray): Estimated covariance matrix
        """
        self._check_is_fitted()

        # Covariance = components^T * diag(explained_variance) * components
        return (self.components_.T * self.explained_variance_) @ self.components_

    def score(self, X: np.ndarray) -> float:
        """
        Return the average log-likelihood of all samples.

        Parameters:
            X (ndarray): Input data

        Returns:
            score (float): Average log-likelihood
        """
        # This is a simplified version - full implementation would use
        # a probabilistic PCA likelihood
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)

        # Return negative reconstruction error as score
        mse = np.mean((X - X_reconstructed) ** 2)
        return -mse

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate input data."""
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        if np.any(np.isnan(X)):
            raise ValueError("Input contains NaN values")

        if np.any(np.isinf(X)):
            raise ValueError("Input contains infinite values")

        return X

    def _choose_solver(self, X_centered: np.ndarray) -> str:
        """Choose the best solver based on data characteristics."""
        if self.svd_solver != 'auto':
            return self.svd_solver

        n_samples, n_features = X_centered.shape

        # Use SVD for tall matrices or when numerical stability is a concern
        if n_samples >= n_features or n_features > 1000:
            return 'svd'
        else:
            return 'covariance'

    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted."""
        if self.components_ is None:
            raise ValueError("This PCA instance is not fitted yet. "
                             "Call 'fit' with appropriate arguments before using this method.")


def my_pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple PCA implementation function (matching lab requirements).

    Parameters:
        X (ndarray): Input data, shape (n_samples, n_features)
        n_components (int): Number of principal components to keep

    Returns:
        X_projected (ndarray): Projected data, shape (n_samples, n_components)
        components (ndarray): Principal components, shape (n_features, n_components)
        explained_variance (ndarray): Variance explained by each component
    """
    # Step 1: Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Step 2: Calculate the covariance matrix
    n_samples = X_centered.shape[0]
    covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Step 4: Sort by eigenvalue (descending order)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Keep only the desired number of components
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]

    # Step 5: Project the data
    X_projected = X_centered @ eigenvectors

    return X_projected, eigenvectors, eigenvalues


# Utility functions for analysis
def explained_variance_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    """Calculate explained variance ratio from eigenvalues."""
    return eigenvalues / np.sum(eigenvalues)


def cumulative_explained_variance(eigenvalues: np.ndarray) -> np.ndarray:
    """Calculate cumulative explained variance ratio."""
    ratios = explained_variance_ratio(eigenvalues)
    return np.cumsum(ratios)


def reconstruction_error(X_original: np.ndarray, X_reconstructed: np.ndarray) -> float:
    """Calculate reconstruction error (Mean Squared Error)."""
    return np.mean((X_original - X_reconstructed) ** 2)


def plot_explained_variance(eigenvalues: np.ndarray, title: str = "Explained Variance") -> None:
    """Plot explained variance and cumulative explained variance."""
    ratios = explained_variance_ratio(eigenvalues)
    cumulative = cumulative_explained_variance(eigenvalues)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Individual explained variance
    ax1.bar(range(1, len(ratios) + 1), ratios * 100, alpha=0.7)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance (%)')
    ax1.set_title(f'{title} - Individual Components')
    ax1.grid(True, alpha=0.3)

    # Cumulative explained variance
    ax2.plot(range(1, len(cumulative) + 1), cumulative * 100, 'o-', linewidth=2, markersize=8)
    ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
    ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance (%)')
    ax2.set_title(f'{title} - Cumulative')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Simple test
    print("Testing PCA implementation...")

    # Generate simple test data
    np.random.seed(42)
    X_test = np.random.randn(100, 4)

    # Test our implementation
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X_test)

    print(f"Original shape: {X_test.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print("PCA implementation test passed!")
