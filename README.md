# Email Spam Classifier

**A Machine Learning based solution to detect and filter malicious emails with high precision.**

## Project Overview

In the modern digital era, spam emails are more than just a nuisance, they are a security risk. This project uses **Natural Language Processing (NLP)** and Machine Learning to classify emails as **Spam** or **Ham** (Legitimate).

### Key Features

* **Real-time Prediction:** Classify text inputs instantly.
* **NLP Pipeline:** Implementation of Tokenization, Stop-word removal, and Stemming.
* **Vectorization:** Uses TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction.
* **High Accuracy:** Optimized using the Naive Bayes / MultinomialNB algorithm.

## Tech Stack

* **Language:** Python
* **Libraries:** Scikit-learn, Pandas, NLTK
* **Deployment:** Streamlit

## How It Works

1. **Data Collection:** Trained on the SMS Spam Collection dataset.
2. **Preprocessing:** Cleaning raw text by removing punctuation and converting to lowercase.
3. **Feature Engineering:** Converting text into numerical vectors using TF-IDF.
4. **Model Training:** Training a Multinomial Naive Bayes model for classification.

## Performance Metrics

`Model Accuracy: 97.77 %`

`Classification Report:`

| | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- | 
| **ham** | 0.98 | 1.00 | 0.99 | 904 |
| **spam** | 0.99 | 0.83 | 0.90 | 128 |
| **accuracy** | | |0.98 | 1032 |
| **macro avg** | 0.98 | 0.91 | 0.94 | 1032 |
| **weighted avg** | 0.98 | 0.98 | 0.98 | 1032 |

> **Result Explanation:**
 The spam detection model achieved an accuracy of 97.77%, which means it correctly classified most of the emails as spam or ham. The precision and recall values show that the model is very effective at identifying ham emails with high accuracy, while also performing strongly in detecting spam emails Overall, the results indicate that the model is reliable and  suitable for real-world email spam filtering applications.

## Installation & Usage

1. **Clone the repository:**

```bash
git clone https://github.com/alisulmanpro/Email-Classifier-1.git

```


2. **Install dependencies:**

```bash
pip install -r requirements.txt
```


3. **Run the application:**

```bash
streamlit run main.py

```

## Spam Testing
<img width="1352" height="644" alt="Image" src="https://github.com/user-attachments/assets/8c1ae09f-939a-43de-8c5b-b71c695f9813" />

## Ham Testing
<img width="1345" height="652" alt="Image" src="https://github.com/user-attachments/assets/11943d71-fb9c-4013-bf05-f649f3847a7e" />

## Common words in spam email
<img width="527" height="288" alt="Image" src="https://github.com/user-attachments/assets/d6a4a8c1-0bbb-4640-bf02-7312639c6c73" />

## Common words in ham email
<img width="528" height="288" alt="Image" src="https://github.com/user-attachments/assets/03ec3bf2-71b6-4be9-9905-5019444750a8" />

## Confusion Matrix
<img width="436" height="365" alt="Image" src="https://github.com/user-attachments/assets/f4662d58-4f19-4d83-a958-5d39e37c3138" />
