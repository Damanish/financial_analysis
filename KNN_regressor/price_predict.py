import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

# Data loading and preprocessing
df = pd.read_csv('data/All_new.csv')
df = df.loc[:, "BIDU_Open":"BIDU_Adj Close"]  #taking the data for amazon
df.insert(0, 'Day', range(1, 1 + len(df)))  # Add "Day" column
df['Average'] = (df['BIDU_High'] + df['BIDU_Low'])/2

# Calculate rolling average
Rolling_avg = []
for i in range(len(df)):
    if i<10:
        Rolling_avg.append(df['Average'][i])
    else:
        Rolling_avg.append(np.mean(df['Average'][i-10:i]))
Rolling_avg = np.array(Rolling_avg)
df['Rolling_Avg'] = Rolling_avg

print(df.head())
print(df.isnull().sum())  # it is coming 0 so no null values present

X = df[['Day', 'Rolling_Avg']]
y = df['Average']

#function to standardize the data
def Standard_Scalar(data):
    data = data.copy()
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean)/std
    return data

#standardizing the rolling_avg column and day column
X_cleaned_Data = {'Day': Standard_Scalar(df['Day']), 'Rolling_Avg': Standard_Scalar(df['Rolling_Avg'])}
X_cleaned = pd.DataFrame(X_cleaned_Data)

#taking last 10 days as test using the values of previous days
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=200,shuffle = False)

class KNNRegressor:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        predictions = []

        for row in X:
            # Compute distances from row to all training points
            distances = np.linalg.norm(self.X_train - row, axis=1)

            # Get the indices of the k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            # Get the target values of the k nearest neighbors
            k_neighbor_values = self.y_train[k_indices]

            # weighted prediction based on inverse distance
            weights = 1 / (distances[k_indices] + 1e-8)
            prediction = np.dot(k_neighbor_values, weights) / np.sum(weights)

            predictions.append(prediction)

        return np.array(predictions)

#function to find mean square error
def mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)  # Convert to NumPy arrays
    return np.mean((y_true - y_pred) ** 2)  # Compute MSE

# Train the model
reg = KNNRegressor(k=3)
reg.fit(X_train, y_train)

# Evaluate
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
mse_value = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error:", mse_value)

# Plot training data
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(len(y_train)), y_train, label="Actual Training", alpha=0.7, marker='o')
plt.plot(range(len(y_train)), y_train_pred, label="Predicted Training", alpha=0.7, marker='x')
plt.title("Training Data Fit")
plt.xlabel("Sample Index")
plt.ylabel("Average Price (avg)")
plt.legend()
plt.grid(True)

# Plot test data
plt.subplot(1, 2, 2)
plt.plot(range(len(y_test)), y_test, label="Actual Y", marker='o', linestyle='dashed', alpha=0.7)
plt.plot(range(len(y_test)), y_test_pred, label="Predicted Y", marker='x', linestyle='solid', alpha=0.7)
plt.xlabel("Test Sample Index")
plt.ylabel("Average Price (avg)")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.grid(True)
plt.show()

# Plot full dataset with predictions
plt.figure(figsize=(15, 6))
y_full = np.concatenate([y_train, y_test])
y_pred_full = np.concatenate([y_train_pred, y_test_pred])

plt.plot(range(len(y_full)), y_full, label="Actual", alpha=0.7, marker='o')
plt.plot(range(len(y_pred_full)), y_pred_full, label="Predicted", alpha=0.7, marker='x')
plt.axvline(x=len(y_train), color='r', linestyle='--', label='Train-Test Split')
plt.title("Full Dataset: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Average Price (avg)")
plt.legend()
plt.grid(True)
plt.show()
