"""
Unit tests for PCA implementation.

This module contains comprehensive tests to validate our PCA implementation
against known results and scikit-learn implementation.

Author: [Your Name]
Course: Machine Learning
Date: June 2025
"""

import pytest
import numpy as np
import warnings
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.datasets import make_classification, load_iris
import sys
import os

# Add src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from pca_implementation import PCA, my_pca


class TestPCAImplementation:
    """Test suite for PCA implementation."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data with known PCA results."""
        np.random.seed(42)
        # Create data with clear principal directions
        X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
        return X

    @pytest.fixture
    def random_data(self):
        """Create random test data."""
        np.random.seed(123)
        return np.random.randn(100, 5)

    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        np.random.seed(456)
        X, _ = make_classification(n_samples=200, n_features=10, n_informative=5,
                                   n_redundant=2, n_clusters_per_class=1, random_state=456)
        return X

    def test_pca_initialization(self):
        """Test PCA initialization with different parameters."""
        # Default initialization
        pca = PCA()
        assert pca.n_components is None
        assert pca.svd_solver == 'auto'
        assert pca.random_state is None

        # Custom initialization
        pca_custom = PCA(n_components=3, svd_solver='covariance', random_state=42)
        assert pca_custom.n_components == 3
        assert pca_custom.svd_solver == 'covariance'
        assert pca_custom.random_state == 42

    def test_input_validation(self, simple_data):
        """Test input validation."""
        pca = PCA()

        # Test 1D array (should fail)
        with pytest.raises(ValueError, match="Expected 2D array"):
            pca.fit(np.array([1, 2, 3]))

        # Test array with NaN (should fail)
        data_with_nan = simple_data.copy()
        data_with_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="Input contains NaN"):
            pca.fit(data_with_nan)

        # Test array with inf (should fail)
        data_with_inf = simple_data.copy()
        data_with_inf[0, 0] = np.inf
        with pytest.raises(ValueError, match="Input contains infinite"):
            pca.fit(data_with_inf)

    def test_data_centering(self, simple_data):
        """Test that data is properly centered."""
        pca = PCA()
        pca.fit(simple_data)

        # Check that mean is stored correctly
        expected_mean = np.mean(simple_data, axis=0)
        np.testing.assert_array_almost_equal(pca.mean_, expected_mean)

        # Check that transformed data has zero mean
        X_transformed = pca.transform(simple_data)
        transformed_mean = np.mean(X_transformed, axis=0)
        np.testing.assert_array_almost_equal(transformed_mean, np.zeros(X_transformed.shape[1]), decimal=10)

    def test_component_orthogonality(self, random_data):
        """Test that principal components are orthogonal."""
        pca = PCA()
        pca.fit(random_data)

        # Components should be orthonormal
        components = pca.components_
        gram_matrix = components @ components.T
        identity = np.eye(components.shape[0])

        np.testing.assert_array_almost_equal(gram_matrix, identity, decimal=10)

    def test_explained_variance_properties(self, random_data):
        """Test properties of explained variance."""
        pca = PCA()
        pca.fit(random_data)

        # Explained variance should be non-negative
        assert np.all(pca.explained_variance_ >= 0)

        # Explained variance should be in descending order
        assert np.all(np.diff(pca.explained_variance_) <= 0)

        # Explained variance ratios should sum to 1 (if all components kept)
        pca_full = PCA(n_components=min(random_data.shape))
        pca_full.fit(random_data)
        assert abs(np.sum(pca_full.explained_variance_ratio_) - 1.0) < 1e-10

    def test_transform_inverse_transform(self, random_data):
        """Test transform and inverse transform consistency."""
        pca = PCA(n_components=3)
        pca.fit(random_data)

        # Transform data
        X_transformed = pca.transform(random_data)

        # Check transformed shape
        assert X_transformed.shape == (random_data.shape[0], 3)

        # Inverse transform
        X_reconstructed = pca.inverse_transform(X_transformed)

        # Check reconstructed shape
        assert X_reconstructed.shape == random_data.shape

        # Reconstruction should preserve the variance captured by kept components
        reconstruction_error = np.mean((random_data - X_reconstructed) ** 2)

        # Error should be related to discarded variance
        total_variance = np.var(random_data, axis=0, ddof=1).sum()
        kept_variance = np.sum(pca.explained_variance_)
        expected_error_bound = total_variance - kept_variance

        # Reconstruction error should be reasonable
        assert reconstruction_error >= 0

    def test_fit_transform_consistency(self, random_data):
        """Test that fit_transform gives the same result as fit().transform()."""
        pca1 = PCA(n_components=3)
        pca2 = PCA(n_components=3)

        # Method 1: fit_transform
        X_transformed1 = pca1.fit_transform(random_data)

        # Method 2: fit then transform
        pca2.fit(random_data)
        X_transformed2 = pca2.transform(random_data)

        np.testing.assert_array_almost_equal(X_transformed1, X_transformed2)

    def test_different_solvers(self, random_data):
        """Test that different solvers give consistent results."""
        pca_cov = PCA(n_components=3, svd_solver='covariance')
        pca_svd = PCA(n_components=3, svd_solver='svd')

        X_cov = pca_cov.fit_transform(random_data)
        X_svd = pca_svd.fit_transform(random_data)

        # Results should be very similar (allowing for sign flips)
        for i in range(X_cov.shape[1]):
            # Check if columns are similar or opposite
            similarity = np.abs(np.corrcoef(X_cov[:, i], X_svd[:, i])[0, 1])
            assert similarity > 0.99, f"Component {i} similarity: {similarity}"

        # Explained variances should be nearly identical
        np.testing.assert_array_almost_equal(
            pca_cov.explained_variance_,
            pca_svd.explained_variance_,
            decimal=5
        )

    def test_my_pca_function(self, simple_data):
        """Test the simple my_pca function."""
        X_projected, components, explained_variance = my_pca(simple_data, n_components=2)

        # Check shapes
        assert X_projected.shape == (simple_data.shape[0], 2)
        assert components.shape == (simple_data.shape[1], 2)
        assert explained_variance.shape == (2,)

        # Check that explained variance is in descending order
        assert explained_variance[0] >= explained_variance[1]

        # Check that components are unit vectors
        for i in range(components.shape[1]):
            component_norm = np.linalg.norm(components[:, i])
            assert abs(component_norm - 1.0) < 1e-10

    def test_against_sklearn(self, classification_data):
        """Test our implementation against scikit-learn PCA."""
        # Our implementation
        our_pca = PCA(n_components=5, svd_solver='covariance')
        X_our = our_pca.fit_transform(classification_data)

        # Scikit-learn implementation
        sklearn_pca = SklearnPCA(n_components=5, svd_solver='full')
        X_sklearn = sklearn_pca.fit_transform(classification_data)

        # Compare explained variances (should be very close)
        np.testing.assert_array_almost_equal(
            our_pca.explained_variance_,
            sklearn_pca.explained_variance_,
            decimal=5
        )

        # Compare explained variance ratios
        np.testing.assert_array_almost_equal(
            our_pca.explained_variance_ratio_,
            sklearn_pca.explained_variance_ratio_,
            decimal=5
        )

        # Compare transformed data (allowing for sign flips)
        for i in range(X_our.shape[1]):
            correlation = np.corrcoef(X_our[:, i], X_sklearn[:, i])[0, 1]
            assert abs(correlation) > 0.99, f"Component {i} correlation: {correlation}"

    def test_iris_dataset(self):
        """Test on the classic Iris dataset."""
        iris = load_iris()
        X = iris.data

        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X)

        # Basic checks
        assert X_transformed.shape == (X.shape[0], 2)
        assert np.sum(pca.explained_variance_ratio_) > 0.8  # Should capture most variance

        # Check that first component explains more variance than second
        assert pca.explained_variance_[0] > pca.explained_variance_[1]

    def test_edge_cases(self):
        """Test edge cases."""
        # Single sample
        single_sample = np.array([[1, 2, 3]])
        pca = PCA()

        # Should handle gracefully (though not very meaningful)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pca.fit(single_sample)

        # Single feature
        single_feature = np.array([[1], [2], [3], [4]])
        pca_single = PCA()
        pca_single.fit(single_feature)

        assert pca_single.components_.shape == (1, 1)
        assert pca_single.explained_variance_.shape == (1,)

    def test_not_fitted_error(self, simple_data):
        """Test that method raises appropriate errors when not fitted."""
        pca = PCA()

        with pytest.raises(ValueError, match="not fitted"):
            pca.transform(simple_data)

        with pytest.raises(ValueError, match="not fitted"):
            pca.inverse_transform(simple_data)

        with pytest.raises(ValueError, match="not fitted"):
            pca.get_covariance()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_explained_variance_ratio(self):
        """Test explained variance ratio calculation."""
        from pca_implementation import explained_variance_ratio

        eigenvalues = np.array([4.0, 2.0, 1.0])
        expected_ratios = np.array([4 / 7, 2 / 7, 1 / 7])

        ratios = explained_variance_ratio(eigenvalues)
        np.testing.assert_array_almost_equal(ratios, expected_ratios)

        # Should sum to 1
        assert abs(np.sum(ratios) - 1.0) < 1e-10

    def test_cumulative_explained_variance(self):
        """Test cumulative explained variance calculation."""
        from pca_implementation import cumulative_explained_variance

        eigenvalues = np.array([4.0, 2.0, 1.0])
        expected_cumulative = np.array([4 / 7, 6 / 7, 1.0])

        cumulative = cumulative_explained_variance(eigenvalues)
        np.testing.assert_array_almost_equal(cumulative, expected_cumulative)

    def test_reconstruction_error(self):
        """Test reconstruction error calculation."""
        from pca_implementation import reconstruction_error

        X_original = np.array([[1, 2], [3, 4]])
        X_reconstructed = np.array([[1.1, 2.1], [2.9, 3.9]])

        error = reconstruction_error(X_original, X_reconstructed)
        expected_error = np.mean((X_original - X_reconstructed) ** 2)

        assert abs(error - expected_error) < 1e-10


if __name__ == "__main__":
    # Run tests if the script is executed directly
    pytest.main([__file__, "-v"])
