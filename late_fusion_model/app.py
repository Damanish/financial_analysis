import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

companies = ['AAPL', 'ADBE', 'AMZN', 'BIDU', 'CSCO', 'EA', 'GOOG', 'IBM', 'INTC', 'MSFT', 'NTES', 'NVDA', 'ORCL', 'QCOM', 'TXN']


class FusionModel:
    def __init__(self, ohlc_dim, sentiment_dim, output_dim, hidden_dim):
        self.w1 = np.random.randn(ohlc_dim + sentiment_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((1, hidden_dim))
        self.w3 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b3 = np.zeros((1, output_dim))

        self.ohlc_dim = ohlc_dim
        self.sentiment_dim = sentiment_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivate(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = np.tanh(self.z2)  # Using tanh here for better gradient flow

        self.z3 = np.dot(self.a2, self.w3) + self.b3

        return self.z3

    def loss_fn(self, y_pred, y_true):
        return np.mean(np.square(y_true - y_pred))

    def backward(self, x, y_true, y_pred, lr):
        n = x.shape[0]
        d_loss = 2 * (y_pred - y_true) / n

        d_w3 = np.dot(self.a2.T, d_loss)
        d_b3 = np.sum(d_loss, axis=0, keepdims=True)

        d_hidden_2 = np.dot(d_loss, self.w3.T) * (1 - np.tanh(self.z2) ** 2)  # tanh derivative
        d_w2 = np.dot(self.a1.T, d_hidden_2)
        d_b2 = np.sum(d_hidden_2, axis=0, keepdims=True)

        d_hidden_1 = np.dot(d_hidden_2, self.w2.T) * self.relu_derivate(self.z1)
        d_w1 = np.dot(x.T, d_hidden_1)
        d_b1 = np.sum(d_hidden_1, axis=0, keepdims=True)

        self.w1 -= lr * d_w1
        self.b1 -= lr * d_b1
        self.w2 -= lr * d_w2
        self.b2 -= lr * d_b2
        self.w3 -= lr * d_w3
        self.b3 -= lr * d_b3

        # Debug: Gradient norm check
        # print("Grad norm w3:", np.linalg.norm(d_w3))

    def train(self, ohlc_vector, sentiment, y, epochs, lr):
        x = np.concatenate((ohlc_vector, sentiment), axis=1)

        # Normalize inputs
        x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)

        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = self.loss_fn(y_pred, y)
            self.backward(x, y, y_pred, lr)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, ohlc_vector, sentiment):
        x = np.concatenate((ohlc_vector, sentiment), axis=1)
        x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)  # apply same normalization
        return self.forward(x)

import os
import sys

# Ensure compatibility if __file__ is undefined (e.g., Streamlit)
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))


from SentimentAnalysis.sentiment_analysis import preprocess, RandomForestClassifier
from ohlc_regression.regression import polynomial_features, StandardScaler, gradient_descent, predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error

# Load dataset
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
csv_path = os.path.join(current_dir, '..', 'data', 'All_new.csv')

data = pd.read_csv(csv_path)

# Use consistent data segment
data = data[1000:1700].copy()

# Sentiment preprocessing
data['text'] = data['Top1'] + " " + data['Top2'] + " " + data['Top3'] + " " + data['Top4'] + " " + data['Top5']
data['cleaned'] = data['text'].apply(preprocess)

# TF-IDF on sentiment text
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), min_df=5)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned'])
X_sent = tfidf_matrix.toarray().astype(np.float32)
y_sent = data['Label'].astype(np.int8).values
X_sent_train, X_sent_test = X_sent[:500], X_sent[500:700]
y_sent_train, y_sent_test = y_sent[:500], y_sent[500:700]

# Train sentiment model
rf = RandomForestClassifier(n_trees=200, max_depth=10)
rf.fit(X_sent_train, y_sent_train)
sentiment_output = rf.predict(X_sent_test).reshape(-1, 1)


def train_and_predict_fusion_model(company):
    try:
        st.write(f"### 🔍 Predicting for {company}")
        global data, sentiment_output

        data = data.copy()
        data["avg"] = (data[f"{company}_High"] + data[f"{company}_Low"]) / 2
        data["rolling_avg_10d"] = data["avg"].rolling(window=10).mean()
        data["rolling_avg_10d"] = data["rolling_avg_10d"].fillna(data["avg"])

        # Use 700 data points: first 500 for train, next 200 for test
        start_index = 0
        end_index = 700

        if end_index > len(data):
            raise ValueError("Not enough data for the selected company.")

        dates = np.arange(start_index, end_index).reshape(-1, 1)
        rolling_avg = data["rolling_avg_10d"].values[start_index:end_index].reshape(-1, 1)
        y_reg = data["avg"].values[start_index:end_index]

        X_reg = np.hstack([dates, rolling_avg])
        X_train_reg, X_test_reg = X_reg[:500], X_reg[500:]
        y_train_reg, y_test_reg = y_reg[:500], y_reg[500:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reg)
        X_test_scaled = scaler.transform(X_test_reg)

        degree = 10
        X_poly_train = polynomial_features(X_train_scaled, degree)
        X_poly_test = polynomial_features(X_test_scaled, degree)

        w0 = np.zeros(X_poly_train.shape[1])
        b0 = 0
        w_final, b_final, _ = gradient_descent(
            X_poly_train, y_train_reg, w0, b0,
            *X_poly_train.shape,
            iterations=20000,
            alpha=0.001
        )

        ohlc_output = predict(X_poly_test, w_final, b0).reshape(-1, 1)
        mean_shift = np.mean(y_test_reg - ohlc_output)
        ohlc_output += mean_shift

        reg_mse = mean_squared_error(y_test_reg, ohlc_output)
        y_fusion = y_test_reg.reshape(-1, 1)

        # Fusion model
        x_fusion = np.concatenate((ohlc_output, sentiment_output), axis=1)
        std = np.std(x_fusion, axis=0)
        std[std == 0] = 1e-8
        x_fusion = (x_fusion - np.mean(x_fusion, axis=0)) / std

        fusion_model = FusionModel(ohlc_dim=1, sentiment_dim=1, output_dim=1, hidden_dim=32)
        for epoch in range(1000):
            y_pred = fusion_model.forward(x_fusion)
            loss = fusion_model.loss_fn(y_pred, y_fusion)
            if np.isnan(loss):
                raise ValueError("NaN loss during fusion model training.")
            fusion_model.backward(x_fusion, y_fusion, y_pred, lr=0.01)

        fusion_preds = fusion_model.forward(x_fusion)
        fusion_mse = mean_squared_error(y_fusion, fusion_preds)

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test_reg, label='Ground Truth', linewidth=2)
        ax.plot(ohlc_output, label='Regression', linestyle='--')
        ax.plot(fusion_preds, label='Fusion', linestyle='-.')
        ax.set_title(f"{company} | Regression MSE: {reg_mse:.4f}, Fusion MSE: {fusion_mse:.4f}")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        return reg_mse, fusion_mse, fig

    except Exception as e:
        st.error(f"Prediction failed for {company}: {e}")
        return None, None, None


import streamlit as st

# Sidebar or selectbox
company_map = {
    'Apple Inc.': 'AAPL',
    'Adobe Inc.': 'ADBE',
    'Amazon.com, Inc.': 'AMZN',
    'Baidu, Inc.': 'BIDU',
    'Cisco Systems, Inc.': 'CSCO',
    'Electronic Arts Inc.': 'EA',
    'Alphabet Inc. (Google)': 'GOOG',
    'International Business Machines Corporation (IBM)': 'IBM',
    'Intel Corporation': 'INTC',
    'Microsoft Corporation': 'MSFT',
    'NetEase, Inc.': 'NTES',
    'NVIDIA Corporation': 'NVDA',
    'Oracle Corporation': 'ORCL',
    'QUALCOMM Incorporated': 'QCOM',
    'Texas Instruments Incorporated': 'TXN'
}
company_full_names = list(company_map.keys())
selected_company_name = st.selectbox("Select Company", company_full_names)
ticker = company_map[selected_company_name]

if st.button('Predict Probability'):
    reg_mse, fusion_mse, plot = train_and_predict_fusion_model(ticker)
    
    if reg_mse is not None:
        st.write(f"**Regression MSE:** {reg_mse:.4f}")
        st.write(f"**Fusion MSE:** {fusion_mse:.4f}")
        st.pyplot(plot)

    