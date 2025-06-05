"""
Kernel Principal Component Analysis (KPCA) Implementation

This module implements Kernel PCA for nonlinear dimensionality reduction,
extending our standard PCA implementation to handle nonlinear relationships.

Author: Nelson Masbayi
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Union, Tuple
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import KernelCenterer
import warnings


class KernelPCA:
    """
    Kernel Principal Component Analysis implementation.

    Kernel PCA applies PCA in a high-dimensional feature space induced by a kernel function,
    allowing for nonlinear dimensionality reduction while maintaining the interpretability
    of principal component analysis.

    Mathematical Foundation:
    Instead of analyzing X directly, we analyze φ(X) where φ is an implicit mapping
    to a higher-dimensional space. The kernel trick allows us to compute inner products
    in this space without explicitly computing φ(X).

    K(x_i, x_j) = <φ(x_i), φ(x_j)>

    Attributes:
        n_components (int): Number of components to keep
        kernel (str or callable): Kernel function to use
        gamma (float): Kernel coefficient for rbf, poly and sigmoid kernels
        degree (int): Degree of polynomial kernel
        coef0 (float): Independent term in polynomial and sigmoid kernels
        alpha_ (ndarray): Eigenvectors of the centered kernel matrix
        lambdas_ (ndarray): Eigenvalues of the centered kernel matrix
        X_fit_ (ndarray): Training data used for fitting
        K_fit_rows_ (ndarray): Training kernel matrix rows for transform
        centerer_ (KernelCenterer): Kernel centerer object
    """

    def __init__(self,
                 n_components: int = 2,
                 kernel: Union[str, Callable] = 'rbf',
                 gamma: Optional[float] = None,
                 degree: int = 3,
                 coef0: float = 1.0,
                 alpha: float = 1.0,
                 eigen_solver: str = 'auto',
                 max_iter: Optional[int] = None,
                 remove_zero_eig: bool = False):
        """
        Initialize Kernel PCA.

        Parameters:
            n_components: Number of components to keep
            kernel: Kernel function ('linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed')
            gamma: Kernel coefficient for rbf, poly and sigmoid kernels
            degree: Degree of polynomial kernel
            coef0: Independent term in polynomial and sigmoid kernels
            alpha: Hyperparameter of the ridge regression for learning the inverse transform
            eigen_solver: Eigenvalue solver ('auto', 'dense', 'arpack')
            max_iter: Maximum number of iterations for arpack solver
            remove_zero_eig: Whether to remove components with zero eigenvalues
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.eigen_solver = eigen_solver
        self.max_iter = max_iter
        self.remove_zero_eig = remove_zero_eig

        # Attributes set during fitting
        self.alpha_ = None
        self.lambdas_ = None
        self.X_fit_ = None
        self.K_fit_rows_ = None
        self.centerer_ = None

    def _get_kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute kernel matrix between X and Y.

        Parameters:
            X: Input data
            Y: Optional second input data. If None, compute K(X, X)

        Returns:
            Kernel matrix
        """
        if callable(self.kernel):
            params = {}
        else:
            params = {
                'gamma': self.gamma,
                'degree': self.degree,
                'coef0': self.coef0
            }

        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def _center_kernel_matrix(self, K: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Center the kernel matrix.

        The centering ensures that the mapped data has zero mean in feature space:
        K_centered = K - 1_n K - K 1_n + 1_n K 1_n

        Parameters:
            K: Kernel matrix
            fit: Whether to fit the centerer or use existing one

        Returns:
            Centered kernel matrix
        """
        if fit:
            self.centerer_ = KernelCenterer()
            return self.centerer_.fit_transform(K)
        else:
            return self.centerer_.transform(K)

    def _solve_eigen_problem(self, K_centered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the eigenvalue problem for the centered kernel matrix.

        Parameters:
            K_centered: Centered kernel matrix

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        n_samples = K_centered.shape[0]

        if self.eigen_solver == 'auto':
            if n_samples > 200 and self.n_components < 10:
                solver = 'arpack'
            else:
                solver = 'dense'
        else:
            solver = self.eigen_solver

        if solver == 'dense':
            # Use full eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

            # Sort in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        elif solver == 'arpack':
            from scipy.sparse.linalg import eigsh

            # Use sparse eigen decomposition for large matrices
            try:
                eigenvalues, eigenvectors = eigsh(
                    K_centered,
                    k=self.n_components,
                    which='LA',  # Largest algebraic eigenvalues
                    maxiter=self.max_iter
                )

                # Sort in descending order
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

            except Exception as e:
                warnings.warn(f"ARPACK failed: {e}. Falling back to dense solver.")
                return self._solve_eigen_problem(K_centered)

        else:
            raise ValueError(f"Unknown eigen_solver: {solver}")

        # Remove components with zero or negative eigenvalues if requested
        if self.remove_zero_eig:
            non_zero_idx = eigenvalues > 1e-12
            eigenvalues = eigenvalues[non_zero_idx]
            eigenvectors = eigenvectors[:, non_zero_idx]

        # Keep only the requested number of components
        eigenvalues = eigenvalues[:self.n_components]
        eigenvectors = eigenvectors[:, :self.n_components]

        return eigenvalues, eigenvectors

    def fit(self, X: np.ndarray) -> 'KernelPCA':
        """
        Fit the Kernel PCA model to the data.

        Parameters:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self: Returns the fitted estimator
        """
        X = np.asarray(X)

        # Store training data for transform
        self.X_fit_ = X.copy()

        # Set default gamma for RBF kernel
        if self.kernel == 'rbf' and self.gamma is None:
            self.gamma = 1.0 / X.shape[1]

        # Compute kernel matrix
        K = self._get_kernel(X)

        # Center the kernel matrix
        K_centered = self._center_kernel_matrix(K, fit=True)

        # Solve eigenvalue problem
        self.lambdas_, self.alpha_ = self._solve_eigen_problem(K_centered)

        # Normalize eigenvectors
        # Each eigenvector alpha_i should satisfy: ||alpha_i||^2 * lambda_i = 1
        for i in range(len(self.lambdas_)):
            if self.lambdas_[i] > 1e-12:
                self.alpha_[:, i] = self.alpha_[:, i] / np.sqrt(self.lambdas_[i])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to the kernel PCA space.

        For new data point x, the projection onto the i-th component is:
        y_i = sum_j alpha_ij * K(x_j, x)

        Parameters:
            X: Data to transform of shape (n_samples, n_features)

        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        if self.alpha_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X)

        # Compute kernel matrix between X and training data
        K = self._get_kernel(X, self.X_fit_)

        # Center using the fitted centerer
        K_centered = self._center_kernel_matrix(K, fit=False)

        # Project onto principal components
        X_transformed = K_centered @ self.alpha_

        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and transform the data.

        Parameters:
            X: Training data of shape (n_samples, n_features)

        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Approximate inverse transformation (pre-image problem).

        Note: This is an approximation since the exact inverse may not exist.
        Uses a simple linear approximation in the original space.

        Parameters:
            X_transformed: Data in kernel PCA space

        Returns:
            Approximate reconstruction in original space
        """
        if self.X_fit_ is None:
            raise ValueError("Model has not been fitted yet.")

        warnings.warn("Inverse transform for Kernel PCA is an approximation. "
                      "The exact pre-image may not exist.")

        # Simple linear approximation
        # This is a basic implementation - more sophisticated methods exist
        weights = X_transformed @ self.alpha_.T

        # Weighted combination of training samples
        reconstruction = weights @ self.X_fit_
        reconstruction /= np.sum(weights, axis=1, keepdims=True)

        return reconstruction

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """
        Calculate explained variance ratios for each component.

        Returns:
            Explained variance ratios
        """
        if self.lambdas_ is None:
            raise ValueError("Model has not been fitted yet.")

        # For kernel PCA, we use eigenvalues as a proxy for variance
        total_variance = np.sum(np.abs(self.lambdas_))
        if total_variance == 0:
            return np.zeros_like(self.lambdas_)

        return np.abs(self.lambdas_) / total_variance


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    RBF (Gaussian) kernel implementation.

    K(x, y) = exp(-gamma * ||x - y||^2)

    Parameters:
        X: First set of samples
        Y: Second set of samples
        gamma: Kernel parameter

    Returns:
        Kernel matrix
    """
    # Compute squared distances
    X_norm = np.sum(X ** 2, axis=1, keepdims=True)
    Y_norm = np.sum(Y ** 2, axis=1, keepdims=True)
    distances_sq = X_norm + Y_norm.T - 2 * X @ Y.T

    # Apply RBF kernel
    return np.exp(-gamma * distances_sq)


def polynomial_kernel(X: np.ndarray, Y: np.ndarray,
                      degree: int = 3, coef0: float = 1.0) -> np.ndarray:
    """
    Polynomial kernel implementation.

    K(x, y) = (x^T y + coef0)^degree

    Parameters:
        X: First set of samples
        Y: Second set of samples
        degree: Polynomial degree
        coef0: Independent term

    Returns:
        Kernel matrix
    """
    return (X @ Y.T + coef0) ** degree


# Utility functions for visualization and analysis
def plot_kernel_pca_comparison(X: np.ndarray, y: np.ndarray,
                               kernels: list = ['linear', 'rbf', 'poly'],
                               n_components: int = 2,
                               figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Compare different kernel PCA results.

    Parameters:
        X: Input data
        y: Labels for coloring
        kernels: List of kernels to compare
        n_components: Number of components
        figsize: Figure size
    """
    n_kernels = len(kernels)
    fig, axes = plt.subplots(1, n_kernels + 1, figsize=figsize)

    # Plot original data (first 2 features)
    if X.shape[1] >= 2:
        scatter = axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        axes[0].set_title('Original Data\n(First 2 Features)')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=axes[0])
    else:
        axes[0].text(0.5, 0.5, 'Need ≥2 features\nfor visualization',
                     ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Original Data')

    # Plot kernel PCA results
    for i, kernel in enumerate(kernels):
        kpca = KernelPCA(n_components=n_components, kernel=kernel)
        X_kpca = kpca.fit_transform(X)

        scatter = axes[i + 1].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', alpha=0.7)
        axes[i + 1].set_title(f'Kernel PCA\n({kernel.title()} Kernel)')
        axes[i + 1].set_xlabel('1st Principal Component')
        axes[i + 1].set_ylabel('2nd Principal Component')
        plt.colorbar(scatter, ax=axes[i + 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test Kernel PCA implementation
    print("Testing Kernel PCA implementation...")

    # Generate test data
    np.random.seed(42)

    # Create nonlinear data (two moons)
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

    # Test different kernels
    kernels = ['linear', 'rbf', 'poly']

    for kernel in kernels:
        print(f"\nTesting {kernel} kernel:")
        kpca = KernelPCA(n_components=2, kernel=kernel)
        X_transformed = kpca.fit_transform(X)

        print(f"  Original shape: {X.shape}")
        print(f"  Transformed shape: {X_transformed.shape}")
        print(f"  Explained variance ratio: {kpca.explained_variance_ratio_}")

    print("\n✅ Kernel PCA implementation test completed!")
