import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data = pd.read_csv('All_new.csv')
data_Baidu = data.loc[:, "BIDU_Open" : "BIDU_Adj Close"]

X = data_Baidu[:-1]
y = data_Baidu[1:].reset_index(drop=True).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def gaussian_kernel(X, X_i, tau):
    return np.exp(-np.sum((X - X_i) ** 2) / (2 * tau ** 2))

def LWR(X, y, x_query, tau):
    m = len(X)
    W = np.diag([gaussian_kernel(x_query, X[i], tau) for i in range(m)])

    X_b = np.c_[np.ones(m), X] 
    lambda_reg = 0.01 # Regularization parameter
    theta = np.linalg.solve(X_b.T @ W @ X_b + lambda_reg * np.eye(X_b.shape[1]), X_b.T @ W @ y)

    x_query_b = np.array([1] + list(x_query)) 
    return x_query_b @ theta

predictions = []
for i in range(len(X_test)):
    X_query = X_test.iloc[i].values
    pred = LWR(X_train.values, y_train, X_query, tau=15)
    predictions.append(pred)

predictions = np.array(predictions)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")
