"""
Stock Price Prediction using LSTM Neural Network
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Plots a line graph for the given values.
def plot_graph(figsize, values, column_name):
    plt.figure(figsize=figsize)
    values.plot()
    plt.xlabel("Date")
    plt.ylabel(column_name)
    plt.title(f"{column_name} over Time")
    plt.show()

# previous_days define the days which we analyse and (predict previous_days+1)th day stock price prediction
# Prepares time series data for LSTM input.
def prepare_data(data, previous_days):
    x_data, y_data = [], []
    for i in range(previous_days, len(data)):
        x_data.append(data[i - previous_days:i])
        y_data.append(data[i])
    return np.array(x_data), np.array(y_data)

# these is functio for call model to train the the into modle
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
# Inverts MinMax scaling.
def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

# main part

stock_data = pd.read_csv('data/All_new.csv')

# stock closing price graph
plot_graph((15, 5), stock_data['BIDU_Adj Close'], 'Stock Price')

# Calculate Moving Average
stock_data['MA_for_30_days'] = stock_data['BIDU_Adj Close'].rolling(30).mean()

# Plot Moving Average
plot_graph((15, 5), stock_data['MA_for_30_days'], '30-Day Moving Average')
plot_graph((15, 5), stock_data[['BIDU_Adj Close', 'MA_for_30_days']], 'Stock Price vs MA')

# Normalize the data (convert data into 0 to 1)
adj_close = stock_data[['BIDU_Adj Close']].dropna()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(adj_close)

# Prepare training and testing data
previous_days = 30
x_data, y_data = prepare_data(scaled_data, previous_days)

# Reshape for LSTM input
x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)

# Split into training and testing sets
split_idx = int(len(x_data) * 0.8)
x_train, y_train = x_data[:split_idx], y_data[:split_idx]
x_test, y_test = x_data[split_idx:], y_data[split_idx:]

# Build and train the model
model = build_lstm_model((x_train.shape[1], 1))
model.fit(x_train, y_train, batch_size=1, epochs=10)

# Make predictions
predictions = model.predict(x_test)

# Invert scaling
inv_predictions = inverse_transform(scaler, predictions)
inv_y_test = inverse_transform(scaler, y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y_test, inv_predictions))
print(f"Root Mean Squared Error: {rmse:.4f}")

# Prepare for plotting
plot_index = stock_data.index[split_idx + previous_days:]
plotting_df = pd.DataFrame({
    'Actual': inv_y_test.flatten(),
    'Predicted': inv_predictions.flatten()
}, index=plot_index)

# Plot results
plot_graph((15, 5), plotting_df, 'Test Predictions vs Actual')

# Plot complete data with predictions
full_data = pd.concat([adj_close[:split_idx + previous_days], plotting_df], axis=0)
plot_graph((15, 5), full_data, 'Full Stock Data with Predictions')

# Define a tolerance level ( 5% of actual value)
tolerance = 0.05  
# Compute relative errors
relative_errors = np.abs((inv_y_test.flatten() - inv_predictions.flatten()) / inv_y_test.flatten())
# flatten() is used to convert a multi-dimensional array into a 1D array.
# Count how many predictions are within the tolerance
correct_predictions = np.sum(relative_errors <= tolerance)
total_predictions = len(inv_y_test)
# Calculate accuracy
accuracy = correct_predictions / total_predictions * 100
print(f"Prediction Accuracy within ±{int(tolerance * 100)}%: {accuracy:.2f}%")


