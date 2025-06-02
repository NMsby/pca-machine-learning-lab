# Mathematical Derivations for Principal Component Analysis

**Author**: Nelson Masbayi

---

This document provides detailed step-by-step mathematical derivations for Principal Component Analysis, from the optimization problem formulation to the eigen decomposition solution.

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Optimization Framework](#optimization-framework)
3. [Lagrangian Method](#lagrangian-method)
4. [Eigen decomposition Solution](#eigen-decomposition-solution)
5. [Variance Explanation](#variance-explanation)
6. [Multiple Components](#multiple-components)
7. [Geometric Interpretation](#geometric-interpretation)

---

## Problem Formulation

### Goal
Find a linear transformation that projects high-dimensional data onto a lower-dimensional subspace while preserving maximum variance.

### Setup
- **Input data**: $X \in \mathbb{R}^{n \times d}$ (n samples, d features)
- **Centered data**: $\tilde{X} = X - \mathbf{1}\boldsymbol{\mu}^T$ where $\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i$
- **Objective**: Find unit vector $\mathbf{w}_1$ that maximizes variance of projections

### Mathematical Statement
Maximize the variance of projected data:
$$\text{Var}(\tilde{X}\mathbf{w}_1) = \mathbf{w}_1^T \mathbf{C} \mathbf{w}_1$$

Subject to the constraint:
$$\|\mathbf{w}_1\|^2 = \mathbf{w}_1^T\mathbf{w}_1 = 1$$

Where $\mathbf{C}$ is the covariance matrix:
$$\mathbf{C} = \frac{1}{n-1}\tilde{X}^T\tilde{X}$$

---

## Optimization Framework

### Variance of Projected Data

For a projection direction $\mathbf{w}$, the projected data is:
$$\mathbf{z} = \tilde{X}\mathbf{w}$$

The variance of the projected data is:
$$\text{Var}(\mathbf{z}) = \frac{1}{n-1}\sum_{i=1}^{n}(\mathbf{z}_i - \bar{\mathbf{z}})^2$$

Since the original data is centered, the projected data is also centered ($\bar{\mathbf{z}} = 0$):
$$\text{Var}(\mathbf{z}) = \frac{1}{n-1}\sum_{i=1}^{n}\mathbf{z}_i^2 = \frac{1}{n-1}\mathbf{z}^T\mathbf{z}$$

Substituting $\mathbf{z} = \tilde{X}\mathbf{w}$:
$$\text{Var}(\mathbf{z}) = \frac{1}{n-1}(\tilde{X}\mathbf{w})^T(\tilde{X}\mathbf{w}) = \frac{1}{n-1}\mathbf{w}^T\tilde{X}^T\tilde{X}\mathbf{w}$$

Therefore:
$$\text{Var}(\mathbf{z}) = \mathbf{w}^T\mathbf{C}\mathbf{w}$$

---

## Lagrangian Method

### Constrained Optimization Problem
Maximize: $f(\mathbf{w}) = \mathbf{w}^T\mathbf{C}\mathbf{w}$  
Subject to: $g(\mathbf{w}) = \mathbf{w}^T\mathbf{w} - 1 = 0$

### Lagrangian Function
$$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T\mathbf{C}\mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$$

### First-Order Conditions
Taking the gradient with respect to $\mathbf{w}$ and setting to zero:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2\mathbf{C}\mathbf{w} - 2\lambda\mathbf{w} = 0$$

This simplifies to:
$$\mathbf{C}\mathbf{w} = \lambda\mathbf{w}$$

### Eigenvalue Equation
The first-order condition is exactly the eigenvalue equation! This means:
- $\mathbf{w}$ must be an **eigenvector** of the covariance matrix $\mathbf{C}$
- $\lambda$ is the corresponding **eigenvalue**

---

## Eigen decomposition Solution

### Finding the Optimal Solution
From the eigenvalue equation $\mathbf{C}\mathbf{w} = \lambda\mathbf{w}$, we can find the variance:
$$\mathbf{w}^T\mathbf{C}\mathbf{w} = \mathbf{w}^T(\lambda\mathbf{w}) = \lambda\mathbf{w}^T\mathbf{w} = \lambda$$

Since $\|\mathbf{w}\| = 1$, the variance of the projection is exactly the eigenvalue!

### Choosing the Best Eigenvector
To maximize variance, we choose the eigenvector corresponding to the **largest eigenvalue**:
$$\lambda_1 = \max_i \lambda_i$$
$$\mathbf{w}_1 = \mathbf{v}_1 \text{ (eigenvector corresponding to } \lambda_1\text{)}$$

### Complete Eigen decomposition
The covariance matrix can be decomposed as:
$$\mathbf{C} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T$$

Where:
- $\mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_d]$ (orthogonal matrix of eigenvectors)
- $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d)$ (diagonal matrix of eigenvalues)
- $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_d \geq 0$

---

## Variance Explanation

### Total Variance
The total variance in the original data is:
$$\text{Total Variance} = \text{tr}(\mathbf{C}) = \sum_{i=1}^{d}\lambda_i$$

This follows from the trace property: $\text{tr}(\mathbf{C}) = \text{tr}(\mathbf{V}\mathbf{\Lambda}\mathbf{V}^T) = \text{tr}(\mathbf{\Lambda}) = \sum_{i=1}^{d}\lambda_i$

### Explained Variance by Component k
The $k$-th principal component explains variance equal to $\lambda_k$.

**Proportion of variance explained**:
$$\text{Explained Variance Ratio}_k = \frac{\lambda_k}{\sum_{i=1}^{d}\lambda_i}$$

**Cumulative variance explained by first k components**:
$$\text{Cumulative Variance}_k = \frac{\sum_{i=1}^{k}\lambda_i}{\sum_{i=1}^{d}\lambda_i}$$

### Proof of Variance Preservation
The variance of the $k$-th principal component is:
$$\text{Var}(PC_k) = \text{Var}(\tilde{X}\mathbf{v}_k) = \mathbf{v}_k^T\mathbf{C}\mathbf{v}_k = \mathbf{v}_k^T(\lambda_k\mathbf{v}_k) = \lambda_k$$

---

## Multiple Components

### Sequential Optimization
For the second principal component, we solve:
$$\max_{\mathbf{w}_2} \mathbf{w}_2^T\mathbf{C}\mathbf{w}_2$$
Subject to:
- $\|\mathbf{w}_2\| = 1$
- $\mathbf{w}_2^T\mathbf{w}_1 = 0$ (orthogonality constraint)

### General Solution
Using Lagrange multipliers with both constraints leads to the same eigenvalue equation. The orthogonality constraint is automatically satisfied because eigenvectors of a symmetric matrix corresponding to different eigenvalues are orthogonal.

**General result**: The $k$-th principal component is the eigenvector corresponding to the $k$-th largest eigenvalue.

### Projection Matrix
To project data onto the first $k$ principal components:
$$\mathbf{W}_k = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k] \in \mathbb{R}^{d \times k}$$

**Projected data**:
$$\mathbf{Y} = \tilde{X}\mathbf{W}_k \in \mathbb{R}^{n \times k}$$

### Reconstruction
Data can be approximately reconstructed using:
$$\tilde{X}_{\text{reconstructed}} = \mathbf{Y}\mathbf{W}_k^T = \tilde{X}\mathbf{W}_k\mathbf{W}_k^T$$

**Reconstruction error**:
$$\|\tilde{X} - \tilde{X}_{\text{reconstructed}}\|_F^2 = \sum_{i=k+1}^{d}\lambda_i$$

---

## Geometric Interpretation

### Principal Components as Coordinate System
PCA finds a new orthogonal coordinate system where:
1. **First axis** (PC1) points in the direction of maximum variance
2. **Second axis** (PC2) points in the direction of maximum remaining variance, orthogonal to PC1
3. **Subsequent axes** continue this pattern

### Rotation and Scaling
The transformation can be viewed as:
1. **Rotation**: Align data with principal component axes
2. **Scaling**: Eigenvalues determine the "importance" of each axis
3. **Truncation**: Keep only the most important axes

### Matrix Form
The complete PCA transformation:
$$\mathbf{Y} = \tilde{X}\mathbf{V}$$

Where $\mathbf{V}$ is the matrix of eigenvectors (rotation matrix).

### Inverse Transformation
To go back to the original space:
$$\tilde{X} = \mathbf{Y}\mathbf{V}^T$$

For reconstruction with $k < d$ components:
$$\tilde{X}_k = \mathbf{Y}_k\mathbf{V}_k^T$$

Where $\mathbf{Y}_k$ contains only the first $k$ principal components and $\mathbf{V}_k$ contains the corresponding eigenvectors.

---

## Summary

The mathematical derivation shows that PCA's optimization problem has an elegant closed-form solution through eigen decomposition:

1. **Problem**: Maximize variance of projections
2. **Method**: Lagrange multipliers with unit norm constraint  
3. **Solution**: Eigenvectors of covariance matrix
4. **Interpretation**: Eigenvalues = variance explained by each component

This mathematical foundation provides the theoretical justification for all PCA algorithms and applications.

---

*"The beauty of PCA lies in how a complex optimization problem reduces to a standard linear algebra operation - eigen decomposition."*