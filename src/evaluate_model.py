import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("../data/clean_tickets.csv")

X = df['clean_text']
y = df['category']

tfidf = joblib.load("../model/tfidf_vectorizer.pkl")
model = joblib.load("../model/ticket_classifier.pkl")

X_tfidf = tfidf.transform(X)
y_pred = model.predict(X_tfidf)

print("Classification Report:\n")
print(classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)
labels = model.classes_

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=labels,
            yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
