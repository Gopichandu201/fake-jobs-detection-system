import pandas as pd
import joblib
import json
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg') # For saving plots without GUI

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ── 1. Load Dataset ──────────────────────────────────────────────────────────
print("=" * 6print("  FAKE JOB POSTING DETECTION - MODEL TRAINING (JOBLIB)")
print("=" * 60)

df = pd.read_csv("dataset.csv")
print(f"\n✅ Dataset loaded: {len(df)} records")

# ── 2. Clean Data ─────────────────────────────────────────────────────────────
df = df.fillna("")

# Convert labels to numeric (Real=0, Fake=1)
df["label_num"] = df["label"].map({"real": 0, "fake": 1})

# Combine all text features
df["text"] = (
    df["title"].astype(str) + " " +
    df["company"].astype(str) + " " +
    df["location"].astype(str) + " " +
    df["salary"].astype(str) + " " +
    df["description"].astype(str)
)

print("✅ Data cleaned and text features combined")

# ── 3. Text Preprocessing ────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    return " ".join(tokens)

df["processed_text"] = df["text"].apply(preprocess_text)
print("✅ Text preprocessing complete")

# ── 4. Train/Test Split (80/20) ──────────────────────────────────────────────
X = df["processed_text"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Train/Test split: {len(X_train)} train / {len(X_test)} test (80/20)")

# ── 5. TF-IDF Vectorization ──────────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"✅ TF-IDF applied: {X_train_tfidf.shape[1]} features extracted")

# ── 6. Train Naive Bayes Model ───────────────────────────────────────────────
model = MultinomialNB(alpha=0.1)
model.fit(X_train_tfidf, y_train)
print("✅ MultinomialNB model trained")

# ── 7. Evaluate Model ────────────────────────────────────────────────────────
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 60)
print("  MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"  Accuracy  : {accuracy * 100:.2f}%")
print(f"  Precision : {precision * 100:.2f}%")
print(f"  Recall    : {recall * 100:.2f}%")
print(f"  F1-Score  : {f1 * 100:.2f}%")
print("=" * 60)

# Confusion Matrix Image
cm = confusion_matrix(y_test, y_pred)
os.makedirs("static/img", exist_ok=True)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('static/img/confusion_matrix.png')
print("✅ Confusion Matrix saved → static/img/confusion_matrix.png")

# ── 8. Save Model, Vectorizer, and Metrics using Joblib ──────────────────────
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

metrics = {
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "total_features": X_train_tfidf.shape[1],
    "confusion_matrix": cm.tolist()
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Model saved    → model.joblib")
print("✅ Vectorizer saved → vectorizer.joblib")
print("✅ Metrics saved  → metrics.json")
print("\n🎉 Training complete! Ready for Explainable AI.")