# Principal Component Analysis (PCA) Lab

A comprehensive implementation and analysis of Principal Component Analysis for Machine Learning.
This project demonstrates PCA from mathematical foundations to real-world applications.

## 🎯 Project Overview

This repository contains a complete exploration of PCA including:
- **Mathematical foundations** and theoretical derivations
- **From-scratch implementation** using NumPy
- **Scikit-learn applications** on real datasets
- **Data compression** and feature extraction examples
- **Kernel PCA** for nonlinear dimensionality reduction
- **Comprehensive evaluation** and performance analysis

## 🚀 Features

- ✅ Manual PCA implementation with comprehensive testing
- ✅ Interactive Jupyter notebooks with detailed explanations
- ✅ Real-world dataset analysis (Iris, MNIST, Faces)
- ✅ Data compression with quality analysis
- ✅ Classification performance comparison
- ✅ Visualizations and reporting

## 📁 Project Structure

```
pca-machine-learning-lab/
├── notebooks/              # Interactive analysis notebooks
│   ├── 01_mathematical_foundations.ipynb
│   ├── 02_pca_from_scratch.ipynb
│   ├── 03_scikit_learn_implementation.ipynb
│   ├── 04_applications.ipynb
│   └── 05_bonus_kernel_pca.ipynb
├── src/                    # Source code and utilities
│   ├── pca_implementation.py
│   ├── kernel_pca.py
│   ├── data_utils.py
│   └── visualization_utils.py
├── data/                   # Data and results
│   ├── processed/          # Processed datasets
│   └── results/            # Analysis results
├── reports/                # Final report and figures
│   ├── final_report.pdf
│   └── figures/
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## 🔍 Key Results

### Performance Improvements
- **High-dimensional data (>500D)**: 5-10x speed improvement
- **Medium-dimensional data (50-500D)**: 2-5x speed improvement
- **Memory reduction**: 10-50x decrease in memory usage
- **Accuracy**: Often maintained or improved

### Compression Achievements
- **Optimal ratios**: 5-50x compression depending on quality requirements
- **Quality preservation**: >95% correlation with proper component selection
- **Processing speed**: 200+ images/second on standard hardware

### Kernel PCA Insights
- **Nonlinear patterns**: 2-5x better class separation
- **RBF kernel**: Most versatile for unknown patterns
- **Parameter tuning**: Critical for performance (gamma optimization)

## 🛠️ Installation & Usage

### Quick Start
```bash
# Clone repository
git clone https://github.com/NMsby/pca-machine-learning-lab.git
cd pca-machine-learning-lab

# Create environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Usage Examples

#### Basic PCA Implementation
```python
from src.pca_implementation import PCA
import numpy as np

# Generate sample data
X = np.random.randn(100, 10)

# Apply PCA
pca = PCA(n_components=3)
X_transformed = pca.fit_transform(X)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

#### Kernel PCA for Nonlinear Data
```python
from src.kernel_pca import KernelPCA
from sklearn.datasets import make_moons

# Generate nonlinear data
X, y = make_moons(n_samples=200, noise=0.1)

# Apply Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=1.0)
X_kpca = kpca.fit_transform(X)
```

## 📊 Datasets Used

- **Iris Dataset** - Classic 4D botanical measurements
- **MNIST** - Handwritten digit recognition
- **Olivetti Faces** - Facial recognition dataset
- **Synthetic Data** - Custom generated for testing

## 📈 Results Summary

### Dataset Analysis
| Dataset        | Dimensions | Optimal Components | Improvement |
|----------------|------------|--------------------|-------------|
| Iris           | 4          | 2 (95.8% variance) | 1.2x speed  |
| MNIST Digits   | 64         | 15 (90% variance)  | 3.5x speed  |
| Olivetti Faces | 4,096      | 50 (85% variance)  | 8.2x speed  |

### Application Guidelines
| Use Case  | Components         | Compression | Priority    |
|-----------|--------------------|-------------|-------------|
| Real-time | 5-15% of original  | 5-15x       | Speed       |
| Storage   | 15-30% of original | 2-8x        | Compression |
| Analysis  | 30-50% of original | 1-4x        | Quality     |

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome! Please feel free to:
- Report issues or bugs
- Suggest improvements to documentation
- Share interesting use cases or datasets
- Propose additional features or analyses

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Course materials and lab instructions
- Scikit-learn documentation and examples
- Academic papers on PCA methodology
- Open source community tools and datasets

---

**Author**: Nelson Masbayi  
**Email**: [nmsby.dev@gmail.com](mailto:nmsby.dev@gmail.com)  
**Module**: Machine Learning  
**Institution**: [Strathmore University](https://strathmore.edu)  
**Date**: June 2025