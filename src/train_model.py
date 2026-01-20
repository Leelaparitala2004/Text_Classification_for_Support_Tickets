import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

df = pd.read_csv("../data/clean_tickets.csv")

X = df['clean_text']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

os.makedirs("../model", exist_ok=True)
joblib.dump(model, "../model/ticket_classifier.pkl")
joblib.dump(tfidf, "../model/tfidf_vectorizer.pkl")

print("Model training completed")
