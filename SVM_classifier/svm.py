import pandas as pd
import numpy as np
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
data = pd.read_csv('data/All_new.csv')
data = data[500:1000]
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
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = word_tokenize(text, preserve_line=True)  # Fix is here
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

# Linear SVM from scratch
class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # convert labels to {-1, 1}

        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.where(approx >= 0, 1, 0)  # map back to {0, 1}

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Initialize, train, and evaluate the SVM
svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train)

accuracy = svm.score(X_test, y_test)
print("SVM Accuracy:", accuracy)
# Evaluate the model with additional metrics

y_pred = svm.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
