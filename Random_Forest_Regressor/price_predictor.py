import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Data loading and preprocessing
df = pd.read_csv('data/All_new.csv')
df = df.loc[:, "BIDU_Open":"BIDU_Adj Close"]  #taking the data for BIDU
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

#taking last 200 days as test using the values of previous days
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=200,shuffle = False)


#class for decision tree regressor
class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=None, max_features=None):
        # Initialize the decision tree with hyperparameters
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = None  
    
    def fit(self, X, y):
        # Convert pandas DataFrames or Series to numpy arrays if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        self.n_features = X.shape[1]  # Store number of features
        self.tree = self._build_tree(X, y, depth=0)  # Start building tree from root node
    
    def _build_tree(self, X, y, depth):
        # Recursive function to build the decision tree
        n_samples, n_features = X.shape
        
        # Check stopping conditions
        if n_samples < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return np.mean(y)  # Return mean of target values as a leaf prediction
        
        best_split = self._best_split(X, y, n_features)  # Find best feature and threshold to split on
        if best_split is None:
            return np.mean(y)  # If no valid split is found, return mean
        
        # Retrieve indices for left and right splits
        left_idxs, right_idxs = best_split["groups"]
        
        # Recursively build left and right branches
        left_subtree = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_subtree = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        # Return the current node 
        return {"feature_index": best_split["feature_index"],
                "threshold": best_split["threshold"],
                "left": left_subtree,
                "right": right_subtree}
    
    def _best_split(self, X, y, n_features):
        # Find the best feature and threshold to split the data
        best_mse = float("inf")
        best_split = None
        
        # Randomly select subset of features if max_features is specified
        if self.max_features and self.max_features < n_features:
            feature_indices = np.random.choice(range(n_features), 
                                              self.max_features, 
                                              replace=False)
        else:
            feature_indices = range(n_features)
        
        # Iterate through selected features
        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])  # Unique values for splitting
            for threshold in thresholds:
                left_idxs = X[:, feature_index] <= threshold  # Indices for left split
                right_idxs = ~left_idxs  # Indices for right split
                
                # Skip if one of the splits is empty
                if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
                    continue
                
                mse = self._calculate_mse(y[left_idxs], y[right_idxs])  # Calculate impurity
                
                # Update best split if current is better
                if mse < best_mse:
                    best_mse = mse
                    best_split = {"feature_index": feature_index, "threshold": threshold, "groups": (left_idxs, right_idxs)}
        
        return best_split  # Return dictionary with best split info
    
    def _calculate_mse(self, left_y, right_y):
        # Calculate Mean Squared Error for a potential split
        total_y = np.concatenate([left_y, right_y]) 
        total_variance = np.var(total_y) * len(total_y)  
        left_variance = np.var(left_y) * len(left_y)  
        right_variance = np.var(right_y) * len(right_y) 
        return (left_variance + right_variance) / len(total_y)  
    
    def predict(self, X):
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Prediction
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        # Recursively traverse the tree to make a prediction
        if not isinstance(node, dict):
            return node  # Leaf node reached
        
        # Decide whether to go left or right
        if x[node["feature_index"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])


class RandomForestRegressor:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features=None):
        # Initialize the random forest with hyperparameters
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []  # Will store all decision trees
    
    def fit(self, X, y):
        # Convert pandas DataFrames or Series to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        n_features = X.shape[1]
        self.trees = []
        n_samples = X.shape[0]
        
        # Build each tree using bootstrapped samples
        for _ in range(self.n_trees):
            indices = np.random.choice(range(n_samples), size=n_samples, replace=True)  # Bootstrap sampling
            X_sample, y_sample = X[indices], y[indices]  # Create training sample for the tree
            
            # Create and train a decision tree
            tree = DecisionTreeRegressor(
                min_samples_split=self.min_samples_split, 
                max_depth=self.max_depth,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)  # Add tree to the forest
    
    def predict(self, X):
        # Convert pandas DataFrame to numpy array if necessary
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        #predictions
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)  # Return mean prediction across all trees

#function to find mean square error
def mean_squared_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)  # Convert to NumPy arrays
    return np.mean((y_true - y_pred) ** 2)  # Compute MSE

#to tune hyperparameters

# best_estimators = None
# best_mse = 10000000
# best_max_depth = None
# for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
#     for j in [5, 10, 15, 20]:
#         print("loop: ", i)
#         # Train the model
#         reg = RandomForestRegressor(n_trees=i, max_depth=10, min_samples_split=2)
#         reg.fit(X_train, y_train)

#         # Evaluate
#         y_train_pred = reg.predict(X_train)
#         y_test_pred = reg.predict(X_test)
#         mse_value = mean_squared_error(y_test, y_test_pred)
#         print("Mean Squared Error:", mse_value)
#         if(mse_value < best_mse):
#             best_mse = mse_value
#             best_estimators = i
#             best_max_depth = j

#Train the model
reg = RandomForestRegressor(n_trees=5, max_depth=10, min_samples_split=2)
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
