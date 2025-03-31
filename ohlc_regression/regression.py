import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import combinations_with_replacement
<<<<<<< HEAD
import copy
=======
import matplotlib.pyplot as plt
start = 30000
end = start + 500
df = pd.read_csv("ohlc_regression\ohlc.csv")
df["avg"]=(df["high"]+df["low"])/2
df.rename(columns={'Unnamed: 0':'date'},inplace=True)
df = df.iloc[start:end]
print(df.head())
>>>>>>> 9871f12 (Price Predictor implemented using Random Forest)

# Load Dataset
data = pd.read_csv("data/All_new.csv")
data_Baidu = data.loc[:, "BIDU_Open":"BIDU_Adj Close"]
data_Baidu["avg"] = (data_Baidu["BIDU_High"] + data_Baidu["BIDU_Low"]) / 2

# Create 10-day rolling average feature
data_Baidu["rolling_avg_10d"] = data_Baidu["avg"].rolling(window=10).mean()

# Fill NaN values in the rolling average (first 9 days won't have complete data)
data_Baidu["rolling_avg_10d"].fillna(data_Baidu["avg"], inplace=True)

dates = np.arange(len(data_Baidu)).reshape(-1, 1)  # Sequential day numbers
rolling_avg = data_Baidu["rolling_avg_10d"].values.reshape(-1, 1)  # Rolling average

# Combine features: dates and rolling average
X = np.hstack([dates, rolling_avg])
y = data_Baidu["avg"].values  # Target variable: average price

# # Split into Train/Test (keeping time order)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, shuffle=False)

X_train, X_test = X[:500], X[500:700]
y_train, y_test = y[:500], y[500:700]


# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def polynomial_features(X, degree):
    """
    Generate polynomial features and normalize them - key for stability
    This is the crucial improvement from the reference code
    """
    m, n = X.shape
    poly_features = [np.ones(m)]  # Bias term (x^0)
    
    for d in range(1, degree + 1):
        for terms in combinations_with_replacement(range(n), d):
            poly_features.append(np.prod(X[:, terms], axis=1))
    
    poly_matrix = np.vstack(poly_features).T  # Convert to (m, new_n) shape

    # Compute mean and std for normalization
    mean = np.mean(poly_matrix, axis=0)
    std = np.std(poly_matrix, axis=0)
    
    # Avoid division by zero for constant columns
    std[std == 0] = 1  

    # Normalize features - THIS is the key step missing in the original code
    poly_matrix = (poly_matrix - mean) / std  
    return poly_matrix

def compute_cost(X, y, w, b, m):
    """Compute MSE cost function"""
    predictions = np.dot(X, w) + b
    return np.mean((predictions - y) ** 2) / 2

def gradient(X, y, w, b, m, n):
    """Compute gradients for linear regression"""
    predictions = np.dot(X, w) + b
    errors = predictions - y
    dj_dw = np.dot(X.T, errors) / m
    dj_db = np.mean(errors)
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, m, n, iterations, alpha, verbose=True):
    """Perform gradient descent optimization"""
    w = copy.deepcopy(w)
    b = b
    cost_history = []
    
    for i in range(iterations):
        dj_dw, dj_db = gradient(X, y, w, b, m, n)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Track cost periodically
        if verbose and i % 1000 == 0:
            cost = compute_cost(X, y, w, b, m)
            cost_history.append(cost)
            print(f"Iteration {i}: Cost = {cost:.6f}")
            
        # Check for exploding gradients
        if np.isnan(w).any() or np.isnan(b):
            print(f"Stopped at iteration {i}, weights exploded.")
            break
            
    return w, b, cost_history

def predict(X, w, b):
    """Make predictions with the trained model"""
    return np.dot(X, w) + b

def mean_squared_error(y_true, y_pred):
    """Compute MSE loss"""
    return np.mean((y_true - y_pred) ** 2)

# Train Polynomial Regression
poly_degree = 10
X_poly_train = polynomial_features(X_train_scaled, poly_degree)

# Get dimensions
m, n = X_poly_train.shape
print(f"Training with {n} polynomial features")

# Initialize weights and parameters
initial_w = np.zeros(n)
initial_b = 0
iterations = 50000
alpha = 0.001  # Using the successful learning rate from the example code

# Train model
w_fin, b_fin, cost_history = gradient_descent(X_poly_train, y_train, initial_w, initial_b, m, n, iterations, alpha)

# Make predictions
X_poly_test = polynomial_features(X_test_scaled, poly_degree)
y_pred_test = predict(X_poly_test, w_fin, b_fin)
y_pred_train = predict(X_poly_train, w_fin, b_fin)

mean_shift = np.mean(y_test - y_pred_test)
print("Mean shift in prediction:", mean_shift)

y_pred_test += mean_shift


# Evaluate model
mse_loss = mean_squared_error(y_test, y_pred_test)
print(f"Final MSE Loss on Test Set: {mse_loss:.4f}")
print(f"RMSE: {np.sqrt(mse_loss):.4f}")

# Plot training fit
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(y_train, label="Actual Training Data", alpha=0.7)
plt.plot(y_pred_train, label="Model Fit", alpha=0.7)
plt.title("Training Data Fit")
plt.legend()
plt.grid(True)

# Plot test predictions
plt.subplot(2, 1, 2)
plt.plot(y_test, label="Actual Test Data", alpha=0.7)
plt.plot(y_pred_test, label="Predictions", alpha=0.7)
plt.title(f"Test Data Predictions (MSE: {mse_loss:.4f})")
plt.xlabel("Test Sample Index")
plt.ylabel("Average Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot full dataset prediction
plt.figure(figsize=(15, 6))
# Concatenate actual values
y_full = np.concatenate([y_train, y_test])
# Concatenate predictions
y_pred_full = np.concatenate([y_pred_train, y_pred_test])

plt.plot(y_full, label="Actual", alpha=0.7)
plt.plot(y_pred_full, label="Predicted", alpha=0.7)
plt.axvline(x=len(y_train), color='r', linestyle='--', label='Train-Test Split')
plt.title("Full Dataset: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Average Price")
plt.legend()
plt.grid(True)
plt.show()

# Plot learning curve if we have cost history
if cost_history:
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(cost_history) * 1000, 1000), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Learning Curve")
    plt.grid(True)
    plt.show()
