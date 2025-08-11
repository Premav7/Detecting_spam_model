from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

app = FastAPI(title="Spam Mail Detection API")

class EmailText(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(email: EmailText):
    # Transform text
    transformed_text = vectorizer.transform([email.text])
    
    # Predict
    prediction = model.predict(transformed_text)[0]
    label = "Spam" if prediction == 1 else "Not Spam"
    
    return {"prediction": label}
