import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from preprocess import clean_text

def train():
    print("Loading dataset...")
    df = pd.read_csv('dataset.csv')
    df = df.fillna("")
    
    # Combine relevant columns for text features
    print("Combining text fields and cleaning data...")
    df['combined_text'] = df['title'].astype(str) + " " + df['company'].astype(str) + " " + df['salary'].astype(str) + " " + df['description'].astype(str)
    
    # Apply text processing
    df['processed_text'] = df['combined_text'].apply(clean_text)
    
    # Encoding labels: 'real' -> 0, 'fake' -> 1
    df['label_num'] = df['label'].map({'real': 0, 'fake': 1})
    
    X = df['processed_text']
    y = df['label_num']
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Extracting features with TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Naive Bayes Classifier...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    
    print("Evaluating Model...")
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\\n--- Evaluation Metrics ---")
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")
    print(f"--------------------------\\n")
    
    if accuracy < 0.90:
        print("Warning: Accuracy is below 90%.")
    else:
        print("Model generated with greater than 90% accuracy as expected!")
        
    print("Generating Confusion Matrix visualization...")
    os.makedirs("static", exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('static/confusion_matrix.png')
    print("Confusion Matrix saved to static/confusion_matrix.png")
    
    print("Saving the model and vectorizer using pickle...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("DONE! Models have been successfully trained and saved.")

if __name__ == '__main__':
    train()
