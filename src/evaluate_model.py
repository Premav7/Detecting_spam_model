import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Paths
MODEL_PATH = "models/spam_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
TEST_PATH = "data/processed/test.csv"
REPORT_PATH = "models/model_performance_report.txt"

def evaluate():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    df = pd.read_csv(TEST_PATH)
    df['message'] = df['message'].fillna("")

    X_test = vectorizer.transform(df['message'])
    y_test = df['label']

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("📊 Model Evaluation on Test Set:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    with open(REPORT_PATH, "a") as f:
        f.write(f"\n\n=== Evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc:.4f}\n")

if __name__ == "__main__":
    evaluate()
