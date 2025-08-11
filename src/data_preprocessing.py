import pandas  as pd
import re
import os
import string
from sklearn.model_selection import train_test_split

def load_data(file_path:str) -> pd.DataFrame:
    df=pd.read_csv(file_path, encoding='latin-1')
    df=df[['v1', 'v2']]
    df.columns = ['label', 'message']
    print(df.head())
    return df


def clean_text(text: str) -> str:
    """Clean text by removing punctuation, numbers, and lowercasing."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  
    text = re.sub(r"\d+", "", text) 
    text = re.sub(r"\s+", " ", text).strip()  
    return text

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing steps to the dataset."""
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  
    df['message'] = df['message'].apply(clean_text).fillna("")
    df = df[df['message'].str.strip() != ""]  # Remove empty messages

    return df

def split_and_save_data(df: pd.DataFrame, output_dir: str, test_size=0.2, random_state=42):
    """Split into train/test and save to processed folder."""
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=test_size, random_state=random_state, stratify=df['label']
    )

    os.makedirs(output_dir, exist_ok=True)
    train_df = pd.DataFrame({'message': X_train, 'label': y_train})
    test_df = pd.DataFrame({'message': X_test, 'label': y_test})

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train set saved to: {train_path}")
    print(f"Test set saved to: {test_path}")

if __name__ == "__main__":
    raw_data_path = "data/raw/spam.csv"
    processed_dir = "data/processed"

    df = load_data(raw_data_path)

    df = preprocess_data(df)

    split_and_save_data(df, processed_dir)

