import joblib
from src.preprocessing import preprocess_text
from fastapi import FastAPI, HTTPException
from typing import Literal
from pydantic import BaseModel, Field

# 1. Load model & vectorizer
try:
    model = joblib.load("models/spam_classifier.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
except Exception as e:
    raise RuntimeError(f"Model Loading Failed: {str(e)}")

app = FastAPI(
    title="Email Spam Detection API",
    description="Classifies email text as spam or ham using pre-trained ML model.",
    version="1.0.0"
)

class EmailRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        description='Email body or message to classify',
        example="You won $5000! Click here to claim your prize now."
    )

class SpamResponse(BaseModel):
    label: Literal["spam", "ham"]
    confidence: float = Field(..., ge=0, le=1)
    message: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/perdict", response_model=SpamResponse)
async def classify_email(request: EmailRequest):
    try:
        cleaned = preprocess_text(request.text)
        if len(cleaned.split()) < 2:
            raise ValueError("Text too short after cleaning.")

        # Transform with your TF-IDF
        x = tfidf.transform([cleaned])

        # Get probabilities (assumes class 1 = spam)
        probs = model.predict_proba(x)[0]
        spam_prob = probs[1] if len(probs) > 1 else float(probs[0])  # fallback if binary
        label: Literal["ham", "spam"] = "spam" if spam_prob >= 0.5 else "ham"

        return SpamResponse(
            label=label,
            confidence=round(float(spam_prob), 4),
            message=f"Classified as **{label.upper()}** ({round(spam_prob * 100, 1)}% spam confidence)"
        )

    except Exception as exp:
        raise HTTPException(status_code=422, detail=f"Error: {str(exp)}")