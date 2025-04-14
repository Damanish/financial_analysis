import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Download required NLTK datasets
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load and prepare dataset
data = pd.read_csv('/content/All_new.csv')
data = data[:500]  # Slice your desired portion
data['text'] = data['Top1'] + " " + data['Top2'] + " " + data['Top3'] + " " + data['Top4'] + " " + data['Top5']
sentiment = data[['text', 'Label']]

# Custom stopwords (keep negation words)
def get_custom_stopwords():
    default_stopwords = set(stopwords.words("english"))
    negation_words = {"not", "no", "never", "n't"}
    return default_stopwords - negation_words

stop_words = get_custom_stopwords()
lemmatizer = WordNetLemmatizer()

# Text preprocessing
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z09\s]', '', text)
    words = word_tokenize(text, preserve_line=True)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['cleaned'] = data['text'].apply(preprocess)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), min_df=5)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned'])
X = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()).astype(np.float32)
y = pd.DataFrame(data['Label']).astype(np.int8).values.flatten()

# Balance the dataset
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42)

# Euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Weighted KNN Classifier
class WeightedKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Calculate distances from the test point to all training points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Sort distances and get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get the labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        # Calculate weights based on inverse of distances (closer neighbors have more weight)
        k_nearest_distances = np.array([distances[i] for i in k_indices])
        weights = 1 / (k_nearest_distances + 1e-5)  # Add small value to avoid division by zero

        # Calculate the weighted vote (the closer the neighbor, the more influence it has)
        weighted_vote = np.bincount(k_nearest_labels, weights=weights)

        # Return the class with the highest weighted vote
        return np.argmax(weighted_vote)

# Train and evaluate the Weighted KNN model
knn = WeightedKNN(k=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = np.mean(y_pred == y_test)
print("Weighted KNN Accuracy:", accuracy)

# Evaluate the model with additional metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
