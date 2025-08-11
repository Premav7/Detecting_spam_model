import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_processed_data(train_path: str, test_path: str):
    """Load preprocessed train and test datasets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def train_tfidf_vectorizer(train_texts, max_features=5000):
    """Fit a TF-IDF vectorizer on the training messages."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    return vectorizer, X_train_tfidf

def train_model(X_train_tfidf, y_train):
    """Train a Naive Bayes classifier."""
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model

def evaluate_model(model, vectorizer, X_test, y_test):
    """Evaluate model performance."""
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    return metrics

def save_artifacts(model, vectorizer, model_dir="models"):
    """Save model and vectorizer to disk."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "spam_model.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f" Model saved to: {model_path}")
    print(f" Vectorizer saved to: {vectorizer_path}")

if __name__ == "__main__":
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    
    train_df, test_df = load_processed_data(train_path, test_path)

    vectorizer, X_train_tfidf = train_tfidf_vectorizer(train_df['message'])
    model = train_model(X_train_tfidf, train_df['label'])
    metrics = evaluate_model(model, vectorizer, test_df['message'], test_df['label'])
    print("📊 Model Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    save_artifacts(model, vectorizer)
