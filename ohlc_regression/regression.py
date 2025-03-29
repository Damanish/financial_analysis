import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
import numpy as np
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
start = 30000
end = start + 500
df = pd.read_csv("ohlc.csv")
df["avg"]=(df["high"]+df["low"])/2
df.rename(columns={'Unnamed: 0':'date'},inplace=True)
df = df.iloc[start:end]
print(df.head())

X = df[['date']] - start
y = df['avg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


def polynomial_features(X, degree):
    m, n = X.shape
    poly_features = [np.ones(m)]  # Bias term (x^0)
    
    for d in range(1, degree + 1):
        for terms in combinations_with_replacement(range(n), d):
            poly_features.append(np.prod(X[:, terms], axis=1))
    
    poly_matrix = np.vstack(poly_features).T  # Convert to (m, new_n) shape

    # Compute mean and std
    mean = np.mean(poly_matrix, axis=0)
    std = np.std(poly_matrix, axis=0)
    
    # Avoid division by zero for constant columns
    std[std == 0] = 1  

    # Normalize features
    poly_matrix = (poly_matrix - mean) / std  
    return poly_matrix


def compute_cost(X, y, w, b, m):
    predictions = np.dot(X, w) + b
    return np.mean((predictions - y) ** 2) / 2  # MSE / 2 for gradient consistency

def gradient(X, y, w, b, m, n):
    predictions = np.dot(X, w) + b
    errors = predictions - y
    dj_dw = np.dot(X.T, errors) / m  # Vectorized gradient
    dj_db = np.mean(errors)  # Scalar bias gradient
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, m, n, iterations, alpha):
    w = copy.deepcopy(w)
    b = b
    for i in range(iterations):
        dj_dw, dj_db = gradient(X, y, w, b, m, n)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Debugging: Check if values explode
        if np.isnan(w).any() or np.isnan(b):
            print(f"Stopped at iteration {i}, weights exploded.")
            break
    return w, b

def predict(X, w, b):
    """Compute predictions for a given dataset."""
    return np.dot(X, w) + b

def mean_squared_error(y_true, y_pred):
    """Compute the Mean Squared Error (MSE) loss."""
    return np.mean((y_true - y_pred) ** 2)

# --- Training ---
degree = 40
X_poly_train = polynomial_features(X_train_scaled, degree)

m, n = X_poly_train.shape
initial_w = np.zeros(n)
initial_b = 0
iterations = 50000
alpha = 0.1

w_fin, b_fin = gradient_descent(X_poly_train, y_train, initial_w, initial_b, m, n, iterations, alpha)
print("Final weights:", w_fin)
print("Final bias:", b_fin)

# --- Testing ---
X_poly_test = polynomial_features(X_test_scaled, degree)  # Transform test set
y_pred = predict(X_poly_test, w_fin, b_fin)  # Get predictions

# Training data predictions
X_poly_train_for_plot = polynomial_features(X_train_scaled, degree)
y_train_pred = predict(X_poly_train_for_plot, w_fin, b_fin)

plt.subplot(1, 2, 1)
plt.plot(y_train, label="Actual Training", alpha=0.5)
plt.plot(y_train_pred, label="Predicted Training", alpha=0.5)
plt.title("Training Data Fit")
plt.xlabel("Sample Index")
plt.ylabel("Average Price (avg)")
plt.legend()
plt.grid(True)

# Compute final test loss (MSE)
mse_loss = mean_squared_error(y_test, y_pred)
print("Final MSE Loss on Test Set:", mse_loss)

plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual Y", marker='o', linestyle='dashed', alpha=0.7)
plt.plot(y_pred, label="Predicted Y", marker='x', linestyle='solid', alpha=0.7)
plt.xlabel("Test Sample Index")
plt.ylabel("Average Price (avg)")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()


# Plot full dataset with predictions
plt.figure(figsize=(15, 6))
y_full = np.concatenate([y_train, y_test])
y_pred_full = np.concatenate([y_train_pred, y_pred])

plt.plot(y_full, label="Actual", alpha=0.7)
plt.plot(y_pred_full, label="Predicted", alpha=0.7)
plt.axvline(x=len(y_train), color='r', linestyle='--', 
            label='Train-Test Split')
plt.title("Full Dataset: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Average Price (avg)")
plt.legend()
plt.grid(True)
plt.show()