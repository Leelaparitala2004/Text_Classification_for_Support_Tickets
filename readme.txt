# Support Ticket Text Classification

## Project Overview

This project implements a Machine Learning–based **Text Classification system** to automatically categorize customer support tickets into predefined categories such as Login Issue, Payment Issue, Technical Issue, etc.

The system uses **Natural Language Processing (NLP)** techniques and a **TF-IDF + Logistic Regression** model to classify ticket text efficiently.

---

## Technologies Used

* Python 3.9+
* Pandas
* NLTK
* Scikit-learn
* Matplotlib & Seaborn
* Joblib

---

## Project Structure

```
support_ticket_classification/
│
├── data/
│   ├── support_tickets.csv        # Raw dataset
│   └── clean_tickets.csv          # Preprocessed dataset
│
├── model/
│   ├── ticket_classifier.pkl      # Trained ML model
│   └── tfidf_vectorizer.pkl       # TF-IDF vectorizer
│
├── src/
│   ├── preprocess.py              # Text cleaning & preprocessing
│   ├── train_model.py             # Model training script
│   └── evaluate_model.py          # Model evaluation & visualization
│
├── requirements.txt
└── README.md
```

---

## Workflow

1. **Data Ingestion**

   * Load support ticket data from CSV

2. **Text Preprocessing**

   * Lowercasing
   * Removing punctuation
   * Stopword removal
   * Lemmatization

3. **Feature Extraction**

   * TF-IDF Vectorization (unigrams + bigrams)

4. **Model Training**

   * Logistic Regression classifier
   * Train-test split

5. **Evaluation**

   * Precision, Recall, F1-score
   * Confusion Matrix visualization

---

## How to Run the Project

### 1. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing

```bash
python src/preprocess.py
```

### 4. Train the Model

```bash
python src/train_model.py
```

### 5. Evaluate the Model

```bash
python src/evaluate_model.py
```

---

## Output

* Classification Report (Precision, Recall, F1-score)
* Confusion Matrix Heatmap
* Saved trained model and vectorizer

---

## Notes

* This project uses a sample dataset for demonstration purposes.
* The approach can be extended to large-scale real-world datasets.

---

## Author

**Leela Nandan**

---

## License

This project is created for learning and evaluation purposes.
