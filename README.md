# 📈 Stock Market Analysis using Regression, Classification, and Fusion Techniques

This project explores **stock price prediction** using both numerical stock data and textual news headlines. We implemented a **multi-model pipeline** involving polynomial regression, custom random forest classification, and a fusion neural network that combines both data types for enhanced performance. The focus is on **Baidu stock prices** and their related news headlines.

---

## Project Overview

- **Regression Model:** Polynomial Regression (degree=10) for predicting stock prices using OHLC data.
- **Classification Model:** Random Forest Classifier using TF-IDF on news headlines to extract sentiment signals.
- **Fusion Model:** A neural architecture combining both numerical and textual features for superior stock price prediction.

---

## 1. Polynomial Regression (Numerical Data)

### Highlights:
- **MSE:** 24.38
- **RMSE:** 4.94
- Significantly outperformed KNN and Random Forest Regressors.

### Techniques Used:
- Feature Engineering: 10-day rolling average, polynomial terms.
- Custom implementation of gradient descent.
- Visualizations: Training fit, prediction error, full dataset view, learning curve.

---

## 2. Text Classification (News Headlines)

### Highlights:
- **Best Accuracy (Random Forest):** ~72%
- Outperformed SVM and Weighted K-NN.

### Techniques Used:
- Custom preprocessing: Tokenization, Lemmatization, Stopword filtering.
- Feature Extraction: TF-IDF (Max features: 5000).
- Custom-built Random Forest Classifier (200 trees, depth 10, Gini impurity).

---

## 3. Fusion Neural Network (Multimodal)

### Highlights:
- **Fusion Model MSE:** 38.24
- Generalization error improved by combining structured + unstructured data.

### Architecture:
- Input: Concatenated numerical + textual features.
- Hidden Layers: ReLU + Tanh
- Output: Linear price prediction
- Optimizer: Gradient Descent (Manual backprop)

---

## Final Results Summary

| Model Type             | Metric    | Value   |
|------------------------|-----------|---------|
| Polynomial Regression  | Test MSE  | **24.38** |
| Polynomial Regression  | RMSE      | 4.94    |
| Text Classifier        | Accuracy  | ~70%    |
| Fusion Model           | Test MSE  | **38.24** |
