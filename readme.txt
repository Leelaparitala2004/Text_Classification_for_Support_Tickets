# Support Ticket Classification

## Overview

This project classifies customer support tickets into predefined categories using basic NLP and Machine Learning techniques. It is built as part of a technical assignment to demonstrate text preprocessing, feature extraction, model training, and evaluation.

---

## Tech Stack

* Python
* Pandas
* NLTK / spaCy
* Scikit-learn
* Matplotlib

---

## Project Structure

```
support_ticket_classification/
│
├── data/
│   ├── support_tickets.csv
│   └── clean_tickets.csv
│
├── model/
│   ├── ticket_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── requirements.txt
└── README.md
```

---

## Workflow

1. Load ticket data from CSV
2. Clean text (lowercase, remove punctuation, stopwords, lemmatize)
3. Convert text to numerical features using TF-IDF
4. Train a multi-class classifier
5. Evaluate using precision, recall, F1-score, and confusion matrix

---

## How to Run

```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train_model.py
python src/evaluate_model.py
```

---

## Dataset

The dataset is synthetically generated using **Mockaroo** for demonstration purposes.

---

## Output

* Classification report
* Confusion matrix visualization
* Saved trained model files

---

## Author

Leela Nandan
