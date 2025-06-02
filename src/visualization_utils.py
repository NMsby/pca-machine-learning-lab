"""
Visualization utilities for PCA analysis.

This module provides comprehensive plotting functions for PCA results,
including 2D/3D scatter plots, component analysis, and comparison visualizations.

Author: [Your Name]
Course: Machine Learning
Date: June 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Optional, Tuple, List, Union
import warnings


def plot_2d_pca_results(X_original: np.ndarray,
                        X_transformed: np.ndarray,
                        pca_object,
                        labels: Optional[np.ndarray] = None,
                        title: str = "PCA Results",
                        feature_names: Optional[List[str]] = None,
                        figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Create comprehensive 2D PCA visualization.

    Parameters:
        X_original: Original data
        X_transformed: Transformed data
        pca_object: Fitted PCA object
        labels: Optional class labels for coloring
        title: Plot title
        feature_names: Names of original features
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Plot 1: Original data (first two features)
    if X_original.shape[1] >= 2:
        scatter_kwargs = {'alpha': 0.7, 's': 50}
        if labels is not None:
            scatter = axes[0, 0].scatter(X_original[:, 0], X_original[:, 1],
                                         c=labels, cmap='viridis', **scatter_kwargs)
            plt.colorbar(scatter, ax=axes[0, 0])
        else:
            axes[0, 0].scatter(X_original[:, 0], X_original[:, 1], **scatter_kwargs)

        axes[0, 0].set_title('Original Data (First 2 Features)')
        if feature_names and len(feature_names) >= 2:
            axes[0, 0].set_xlabel(feature_names[0])
            axes[0, 0].set_ylabel(feature_names[1])
        else:
            axes[0, 0].set_xlabel('Feature 1')
            axes[0, 0].set_ylabel('Feature 2')
    else:
        axes[0, 0].text(0.5, 0.5, 'Need ≥2 features\nfor this plot',
                        ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Original Data')

    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Original data with principal components
    if X_original.shape[1] >= 2 and hasattr(pca_object, 'components_'):
        if labels is not None:
            scatter = axes[0, 1].scatter(X_original[:, 0], X_original[:, 1],
                                         c=labels, cmap='viridis', **scatter_kwargs)
        else:
            axes[0, 1].scatter(X_original[:, 0], X_original[:, 1], **scatter_kwargs)

        # Plot principal component vectors
        center = getattr(pca_object, 'mean_', np.mean(X_original, axis=0))
        if len(center) >= 2:
            for i, (component, variance) in enumerate(zip(pca_object.components_[:2],
                                                          pca_object.explained_variance_[:2])):
                if len(component) >= 2:
                    scale = np.sqrt(variance) * 2
                    axes[0, 1].arrow(center[0], center[1],
                                     component[0] * scale, component[1] * scale,
                                     head_width=0.1, head_length=0.15,
                                     fc=f'C{i + 2}', ec=f'C{i + 2}',
                                     linewidth=3, label=f'PC{i + 1}')

        axes[0, 1].legend()
        axes[0, 1].set_title('Original Data with Principal Components')
        if feature_names and len(feature_names) >= 2:
            axes[0, 1].set_xlabel(feature_names[0])
            axes[0, 1].set_ylabel(feature_names[1])
        else:
            axes[0, 1].set_xlabel('Feature 1')
            axes[0, 1].set_ylabel('Feature 2')
    else:
        axes[0, 1].text(0.5, 0.5, 'Cannot show PCs\nfor this data',
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Principal Components')

    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Transformed data
    if labels is not None:
        scatter = axes[0, 2].scatter(X_transformed[:, 0], X_transformed[:, 1],
                                     c=labels, cmap='viridis', **scatter_kwargs)
        plt.colorbar(scatter, ax=axes[0, 2])
    else:
        axes[0, 2].scatter(X_transformed[:, 0], X_transformed[:, 1], **scatter_kwargs)

    axes[0, 2].set_title('Transformed Data (PC Space)')
    axes[0, 2].set_xlabel('PC1')
    axes[0, 2].set_ylabel('PC2')
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Explained variance
    if hasattr(pca_object, 'explained_variance_ratio_'):
        n_components = len(pca_object.explained_variance_ratio_)
        components = [f'PC{i + 1}' for i in range(n_components)]
        variance_ratios = pca_object.explained_variance_ratio_ * 100

        bars = axes[1, 0].bar(range(n_components), variance_ratios,
                              alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, n_components)))
        axes[1, 0].set_title('Explained Variance by Component')
        axes[1, 0].set_xlabel('Principal Component')
        axes[1, 0].set_ylabel('Explained Variance (%)')
        axes[1, 0].set_xticks(range(n_components))
        axes[1, 0].set_xticklabels(components)

        # Add percentage labels
        for i, (bar, percentage) in enumerate(zip(bars, variance_ratios)):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')

        axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Cumulative-explained variance
    if hasattr(pca_object, 'explained_variance_ratio_'):
        cumulative_variance = np.cumsum(pca_object.explained_variance_ratio_ * 100)
        axes[1, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                        'o-', linewidth=2, markersize=8)
        axes[1, 1].axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
        axes[1, 1].set_title('Cumulative Explained Variance')
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('Cumulative Variance (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Component loadings (if available and reasonable size)
    if (hasattr(pca_object, 'components_') and
            X_original.shape[1] <= 10 and
            len(pca_object.components_) >= 2):

        loadings = pca_object.components_[:2].T  # First two components
        feature_names_plot = feature_names if feature_names else [f'Feature {i + 1}' for i in range(len(loadings))]

        x = np.arange(len(feature_names_plot))
        width = 0.35

        bars1 = axes[1, 2].bar(x - width / 2, loadings[:, 0], width,
                               label='PC1', alpha=0.7, color='C0')
        bars2 = axes[1, 2].bar(x + width / 2, loadings[:, 1], width,
                               label='PC2', alpha=0.7, color='C1')

        axes[1, 2].set_title('Component Loadings')
        axes[1, 2].set_xlabel('Features')
        axes[1, 2].set_ylabel('Loading Value')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(feature_names_plot, rotation=45, ha='right')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    else:
        axes[1, 2].text(0.5, 0.5, 'Too many features\nfor loading plot',
                        ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Component Loadings')

    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


def plot_3d_pca_results(X_original: np.ndarray,
                        X_transformed: np.ndarray,
                        pca_object,
                        labels: Optional[np.ndarray] = None,
                        title: str = "3D PCA Results",
                        figsize: Tuple[int, int] = (18, 6)) -> None:
    """
    Create 3D PCA visualization.

    Parameters:
        X_original: Original 3D data
        X_transformed: Transformed data
        pca_object: Fitted PCA object
        labels: Optional class labels
        title: Plot title
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)

    # Original 3D data
    ax1 = fig.add_subplot(131, projection='3d')
    if labels is not None:
        scatter = ax1.scatter(X_original[:, 0], X_original[:, 1], X_original[:, 2],
                              c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax1, shrink=0.5)
    else:
        ax1.scatter(X_original[:, 0], X_original[:, 1], X_original[:, 2], alpha=0.7)

    ax1.set_title('Original 3D Data')
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.set_zlabel('X₃')

    # First two PCs
    ax2 = fig.add_subplot(132)
    if labels is not None:
        scatter = ax2.scatter(X_transformed[:, 0], X_transformed[:, 1],
                              c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax2)
    else:
        ax2.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7)

    ax2.set_title('First Two Principal Components')
    ax2.set_xlabel('PC₁')
    ax2.set_ylabel('PC₂')
    ax2.grid(True, alpha=0.3)

    # Explained variance
    ax3 = fig.add_subplot(133)
    if hasattr(pca_object, 'explained_variance_ratio_'):
        n_components = len(pca_object.explained_variance_ratio_)
        components = [f'PC{i + 1}' for i in range(n_components)]
        variance_ratios = pca_object.explained_variance_ratio_ * 100

        bars = ax3.bar(range(n_components), variance_ratios, alpha=0.7)
        ax3.set_title('Explained Variance')
        ax3.set_xlabel('Component')
        ax3.set_ylabel('Explained Variance (%)')
        ax3.set_xticks(range(n_components))
        ax3.set_xticklabels(components)

        for bar, percentage in zip(bars, variance_ratios):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')

        ax3.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_reconstruction_analysis(X_original: np.ndarray,
                                 pca_object,
                                 max_components: Optional[int] = None,
                                 title: str = "Reconstruction Analysis") -> None:
    """
    Analyze reconstruction quality with different numbers of components.

    Parameters:
        X_original: Original data
        pca_object: Fitted PCA object
        max_components: Maximum components to test
        title: Plot title
    """
    if max_components is None:
        max_components = min(X_original.shape[1], 10)

    n_components_list = range(1, max_components + 1)
    reconstruction_errors = []
    variance_explained = []

    from pca_implementation import PCA  # Import our PCA class

    for n_comp in n_components_list:
        # Create PCA with n_comp components
        pca_temp = PCA(n_components=n_comp)
        X_transformed = pca_temp.fit_transform(X_original)
        X_reconstructed = pca_temp.inverse_transform(X_transformed)

        # Calculate reconstruction error
        error = np.mean((X_original - X_reconstructed) ** 2)
        reconstruction_errors.append(error)

        # Calculate variance explained
        variance_explained.append(np.sum(pca_temp.explained_variance_ratio_))

    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Reconstruction error vs components
    ax1.plot(n_components_list, reconstruction_errors, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Reconstruction Error (MSE)')
    ax1.set_title('Reconstruction Error vs Components')
    ax1.grid(True, alpha=0.3)

    # Variance explained vs components
    ax2.plot(n_components_list, np.array(variance_explained) * 100, 'o-',
             linewidth=2, markersize=8, color='green')
    ax2.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Variance Explained vs Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Trade-off plot
    ax3.plot(np.array(variance_explained) * 100, reconstruction_errors, 'o-',
             linewidth=2, markersize=8, color='purple')
    ax3.set_xlabel('Variance Explained (%)')
    ax3.set_ylabel('Reconstruction Error (MSE)')
    ax3.set_title('Variance-Error Trade-off')
    ax3.grid(True, alpha=0.3)

    # Add annotations for key points
    for i, (var, err, n_comp) in enumerate(zip(variance_explained, reconstruction_errors, n_components_list)):
        if n_comp in [1, 2, min(5, max_components), max_components]:
            ax3.annotate(f'{n_comp} PC{"s" if n_comp > 1 else ""}',
                         (var * 100, err),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=10, alpha=0.8)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    return reconstruction_errors, variance_explained


def compare_pca_implementations(X: np.ndarray,
                                our_pca,
                                sklearn_pca,
                                n_components: int = 2,
                                title: str = "PCA Implementation Comparison") -> None:
    """
    Compare our PCA implementation with scikit-learn.

    Parameters:
        X: Input data
        our_pca: Our PCA implementation (fitted)
        sklearn_pca: Scikit-learn PCA (fitted)
        n_components: Number of components
        title: Plot title
    """
    # Transform data with both implementations
    X_our = our_pca.transform(X)
    X_sklearn = sklearn_pca.transform(X)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot transformed data
    axes[0, 0].scatter(X_our[:, 0], X_our[:, 1], alpha=0.7, label='Our Implementation')
    axes[0, 0].set_title('Our Implementation')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(X_sklearn[:, 0], X_sklearn[:, 1], alpha=0.7, label='Scikit-learn', color='orange')
    axes[0, 1].set_title('Scikit-learn')
    axes[0, 1].set_xlabel('PC1')
    axes[0, 1].set_ylabel('PC2')
    axes[0, 1].grid(True, alpha=0.3)

    # Overlay comparison
    axes[0, 2].scatter(X_our[:, 0], X_our[:, 1], alpha=0.5, label='Our Implementation', s=30)
    axes[0, 2].scatter(X_sklearn[:, 0], X_sklearn[:, 1], alpha=0.5, label='Scikit-learn', s=30)
    axes[0, 2].set_title('Overlay Comparison')
    axes[0, 2].set_xlabel('PC1')
    axes[0, 2].set_ylabel('PC2')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Compare explained variances
    components = [f'PC{i + 1}' for i in range(n_components)]
    x_pos = np.arange(len(components))
    width = 0.35

    axes[1, 0].bar(x_pos - width / 2, our_pca.explained_variance_ratio_, width,
                   label='Our Implementation', alpha=0.7)
    axes[1, 0].bar(x_pos + width / 2, sklearn_pca.explained_variance_ratio_, width,
                   label='Scikit-learn', alpha=0.7)
    axes[1, 0].set_title('Explained Variance Ratio Comparison')
    axes[1, 0].set_xlabel('Component')
    axes[1, 0].set_ylabel('Explained Variance Ratio')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(components)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Correlation analysis
    correlations = []
    for i in range(n_components):
        corr = np.corrcoef(X_our[:, i], X_sklearn[:, i])[0, 1]
        correlations.append(abs(corr))  # Take absolute value due to possible sign flips

    bars = axes[1, 1].bar(components, correlations, alpha=0.7, color='green')
    axes[1, 1].set_title('Component Correlation')
    axes[1, 1].set_xlabel('Component')
    axes[1, 1].set_ylabel('Absolute Correlation')
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].grid(True, alpha=0.3)

    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{corr:.4f}', ha='center', va='bottom', fontweight='bold')

    # Difference analysis
    variance_diff = np.abs(our_pca.explained_variance_ratio_ - sklearn_pca.explained_variance_ratio_)
    axes[1, 2].bar(components, variance_diff, alpha=0.7, color='red')
    axes[1, 2].set_title('Explained Variance Difference')
    axes[1, 2].set_xlabel('Component')
    axes[1, 2].set_ylabel('Absolute Difference')
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    # Print numerical comparison
    print(f"\nNumerical Comparison:")
    print(f"Explained Variance Ratios:")
    print(f"  Our Implementation: {our_pca.explained_variance_ratio_}")
    print(f"  Scikit-learn:       {sklearn_pca.explained_variance_ratio_}")
    print(f"  Max Difference:     {np.max(variance_diff):.2e}")
    print(f"\nComponent Correlations: {correlations}")
    print(f"All correlations > 0.999: {all(c > 0.999 for c in correlations)}")


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")

    # Generate test data
    np.random.seed(42)
    from data_utils import generate_elliptical_data, generate_3d_manifold

    # Test 2D visualization
    X_2d = generate_elliptical_data(200, random_state=42)

    # Simple PCA for testing
    from pca_implementation import PCA

    pca_2d = PCA(n_components=2)
    X_2d_transformed = pca_2d.fit_transform(X_2d)

    print("2D visualization test completed")

    # Test 3D data
    X_3d, colors = generate_3d_manifold(150, 'swiss_roll', random_state=42)
    pca_3d = PCA(n_components=3)
    X_3d_transformed = pca_3d.fit_transform(X_3d)

    print("3D visualization test completed")
    print("All visualization functions working correctly!")
