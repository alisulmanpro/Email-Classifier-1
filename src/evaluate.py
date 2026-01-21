import joblib
from src.train import X, y
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load model & vectorizer
model = joblib.load("../models/spam_classifier.pkl")
tfidf = joblib.load("../models/tfidf_vectorizer.pkl")

# 2. Transform text
x = tfidf.transform(X)

# 3. Predict with threshold
threshold = 0.27
y_prob = model.predict_proba(x)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# 4. Evaluation
print("Confusion Matrix")
print(confusion_matrix(y, y_pred))

print("\nClassification Report")
print(classification_report(y, y_pred, target_names=["Ham", "Spam"]))

