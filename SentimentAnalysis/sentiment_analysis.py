import pandas as pd
import numpy as np
import re
import nltk #for text processing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from collections import Counter
from sklearn.model_selection import train_test_split
#enabling required nltk datasets
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

data = pd.read_csv('../data/All_new.csv')
data = data[500:1000]
data['text'] = data['Top1'] + " " + data['Top2'] + " " + data['Top3'] + " " + data['Top4'] + " " + data['Top5']
sentiment = data[['text', 'Label']]

def get_custom_stopwords():
    default_stopwords = set(stopwords.words("english"))
    negation_words = {"not", "no", "never", "n't"}
    return default_stopwords - negation_words

stop_words = get_custom_stopwords()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
  text = str(text).lower()
  text = re.sub(r'[^a-z0-9\s]','', text)
  words = word_tokenize(text)
  words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
  return ' '.join(words)
data['cleaned'] = data['text'].apply(preprocess)

tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), min_df=5)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned'])
X = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()).astype(np.float32)
y = pd.DataFrame(data['Label']).astype(np.int8).values.flatten()

X_majority = X[y == 0]
X_minority = X[y == 1]
y_majority = y[y == 0]
y_minority = y[y == 1]

X_minority_upsampled, y_minority_upsampled = resample(
    X_minority, y_minority,
    replace=True,
    n_samples=len(y_majority),
    random_state=42
)

X_balanced = np.vstack([X_majority, X_minority_upsampled])
y_balanced = np.hstack([y_majority, y_minority_upsampled])

def gini_impurity(y):
  counts = np.bincount(y, minlength=2)
  probabilities = counts / len(y)
  gini = 1 - np.sum(probabilities**2)
  return gini

def split_data(X, y, feature_index, threshold=0):
  left = X[:, feature_index] <= threshold
  right = ~left
  return X[left], y[left], X[right], y[right]

def best_split(X, y, feature_indices):
  best_feature = None
  best_gini = 1
  best_data_split = None

  for feature_index in feature_indices:
    X_left, y_left, X_right, y_right = split_data(X, y, feature_index)
    if len(y_left) == 0 or len(y_right) == 0:
      continue
    gini_left = gini_impurity(y_left)
    gini_right = gini_impurity(y_right)
    gini_split = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)
    if gini_split < best_gini:
      best_gini = gini_split
      best_feature = feature_index
      best_data_split = (X_left, y_left, X_right, y_right)
  return best_feature, best_data_split

class DecisionTree:
  def __init__(self, max_depth):
    self.max_depth = max_depth
    self.tree = None

  def build_tree(self, X, y, depth = 0):
    y = np.array(y).flatten()
    if(len(set(y))==1 or depth == self.max_depth):
      return {"label": np.bincount(y).argmax()}
    num_features = np.sqrt(X.shape[1]).astype(int)
    feature_indices = np.random.choice(range(X.shape[1]), num_features, replace=False)
    best_feature, best_data_split = best_split(X, y, feature_indices)
    if best_feature is None:
      return {"label": np.bincount(y).argmax()}

    X_left, y_left, X_right, y_right = best_data_split
    return {
        "feature": best_feature,
        "left": self.build_tree(X_left, y_left, depth + 1),
        "right": self.build_tree(X_right, y_right, depth + 1)
    }

  def fit(self, X, y):
    self.tree = self.build_tree(X,y)

  def predict_sample(self, sample, node):
    if "label" in node:
      return node["label"]
    if sample[node["feature"]] <= 0:
      return self.predict_sample(sample, node["left"])
    else:
      return self.predict_sample(sample, node["right"])

  def predict(self, X):
    return np.array([self.predict_sample(sample, self.tree) for sample in X])

def bootstrap(X,y):
  n = X.shape[0]
  indices = np.random.choice(n, size=n, replace=True)
  return X[indices], y[indices]

class RandomForestClassifier:
  def __init__(self, n_trees, max_depth):
    self.n_trees = n_trees
    self.max_depth = max_depth
    self.trees = []

  def fit(self, X, y):
    for i in range(self.n_trees):
      X_boot, y_boot = bootstrap(X, y)
      tree = DecisionTree(max_depth=self.max_depth)
      tree.fit(X_boot, y_boot)
      self.trees.append(tree)
      print(f"Tree {i+1} trained.")

  def predict(self, X):
    tree_predictions = np.array([tree.predict(X) for tree in self.trees])
    final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)
    return np.array(final_predictions)

  def score(self, X, y):
    predictions = self.predict(X)
    return np.mean(predictions == y)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, stratify = y_balanced, random_state=42)
rf = RandomForestClassifier(n_trees=200, max_depth=10)
rf.fit(X_train, y_train)

accuracy = rf.score(X_test, y_test)
print("Accuracy:", accuracy)
