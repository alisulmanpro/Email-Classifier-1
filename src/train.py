import joblib
import pandas as pd
from src.preprocessing import preprocess_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

le = LabelEncoder()

df = pd.read_csv('../data/spam.csv', encoding='latin-1') # Load DataSet
df = df.rename(columns={'v1': 'label', 'v2': 'message'}) # Rename Labels

# Drop unnecessary cols
columns_to_drop = [col for col in df.columns if 'Unnamed' in col]
df = df.drop(columns=columns_to_drop)

df['label'] = le.fit_transform(df['label']) # Label Binary Encode

# Remove Duplicate Values
df.drop_duplicates(keep="first", inplace=True)
df.duplicated().sum()

# Insert some cols
df["message_length"] = df.apply(lambda x: len(x["message"]), axis=1)
df["clean_msg"] = df.apply(lambda x: preprocess_text(x["message"]), axis=1)
df["preprocess_length"] = df.apply(lambda x: len(x["clean_msg"]), axis=1)

# Split data for train and test
X = df['clean_msg']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert into numerical form
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    analyzer='word',
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = LogisticRegression(class_weight={0:1, 1:2},max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 8. Save model & vectorizer
joblib.dump(model, "../models/spam_classifier.pkl")
joblib.dump(tfidf, "../models/tfidf_vectorizer.pkl")
