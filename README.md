<img width="1920" height="563" alt="Image" src="https://github.com/user-attachments/assets/dde5d8a0-123c-4d3c-8f3a-84ac56ad59c7" />

# Email/SMS Spam Detection System (Machine Learning)

A production-ready machine learning system to classify **Email/SMS messages** as **Spam** or **Ham (Not Spam)** using classical NLP and supervised learning techniques.

This project is designed with **ML engineering best practices**, focusing on clean architecture, reproducibility, and business-driven evaluation.

---

## Problem Statement

Spam messages cause financial loss, security risks, and poor user experience.  
The goal of this project is to build a **reliable and interpretable spam detection system** that:

- Minimizes **false positives** (important emails marked as spam)
- Maximizes **spam recall**
- Is fast, explainable, and deployable

---

## Dataset

- **Source:** Kaggle – SMS Spam Collection Dataset
- **Type:** Text (Unstructured)
- **Classes:**
  - `Ham` → Legitimate messages
  - `Spam` → Unwanted or malicious messages
- **Challenge:** Class imbalance (more ham than spam)

> Raw dataset is stored without modification for reproducibility.

---

##  Project Structure

``` txt
email-spam-detection/
│
├── data/
│ └── raw_dataset.csv
│
├── notebooks/
│ └── eda_and_training.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── train.py
│ └── evaluate.py
│
├── models/
│ ├── spam_classifier.pkl
│ └── tfidf_vectorizer.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```


---

## Approach Overview

### 1. Text Preprocessing (Minimal, Signal-Preserving)
- Remove HTML tags and email headers
- Normalize whitespace
- Convert text to lowercase
- Replace URLs and email addresses with tokens
- **Keep punctuation signals (`! ? $`)**
- Avoid over-cleaning to preserve spam indicators

### 2. Feature Engineering
- **TF-IDF Vectorization**
- Word n-grams (1–2)
- Sublinear TF scaling
- Rare and overly frequent term filtering

### 3. Model Selection
- **Logistic Regression**
- Class weighting to handle imbalance
- Chosen for:
  - Interpretability
  - Speed
  - Industry adoption

### 4. Decision Threshold Optimization
- Default threshold adjusted from `0.50` → `0.27`
- Improves spam recall while maintaining high precision

---

## Results

### Final Performance (Threshold = 0.27)

| Metric        | Ham     | Spam |
|---------------|---------|------|
| Precision     | 1.00    | 0.94 |
| Recall        | 0.99    | 0.97 |
| F1-Score      | 0.99    | 0.95 |
| Accuracy      | **99%** |

---

## Evaluation Artifacts

### Confusion Matrix & Classification Report
<img width="3840" height="2160" alt="Image" src="https://github.com/user-attachments/assets/0eea3503-46d0-4913-841f-0ad55c9c7856" />

---

##  How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python src/train.py
```

### 3. Evaluate the Model
```bash
python src/evaluate.py
```

## Key Engineering Decisions

- Avoided deep learning due to small dataset size

- Focused on feature engineering over model complexity

- Explicit threshold tuning based on business tradeoffs

- Clean separation between experimentation and production code

## Model Artifacts

Saved in `models/`:

- `spam_classifier.pkl` → Trained Logistic Regression model

- `tfidf_vectorizer.pkl` → TF-IDF feature transformer

## Demo

- Video Demo:

<video src="[./assets/video.mp4](https://github.com/alisulmanpro/email-spam-detection-system/blob/master/assets/video.mp4)" width="100%" controls></video>

## Author

**Ali Sulman** <br/>
Machine Learning Engineer (Aspirant)<br/>
Focused on applied ML, NLP, and production-ready systems.

## License
This project is open for educational and portfolio use.

---

## What This README Does for You

- Looks **professional on GitHub**
- Easy to explain in interviews
- Shows **engineering maturity**
- Ready for deployment extension
