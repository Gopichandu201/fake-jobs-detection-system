# 🛡️ AI-Driven Fake Job Posting Detection System

> **Design and Implementation of an AI-Driven Fake Job Posting Detection System Using Naive Bayes Classification and TF-IDF–Based Natural Language Processing**

## 📋 Overview

An intelligent web application that detects whether a job posting is **REAL** or **FAKE** using Machine Learning. Built with Flask, Naive Bayes (MultinomialNB), and TF-IDF vectorization.

## 🏗️ Folder Structure

```
FakeJobDetector/
├── app.py                  # Flask web application (main backend)
├── train_model.py          # ML model training pipeline
├── dataset.csv             # Training dataset (200+ job postings)
├── model.pkl               # Trained Naive Bayes model
├── vectorizer.pkl          # Fitted TF-IDF vectorizer
├── metrics.json            # Model evaluation metrics
├── database.db             # SQLite database
├── requirements.txt        # Python dependencies
├── .env                    # Environment config (SMTP)
├── static/
│   └── css/
│       └── style.css       # Premium dark theme CSS
└── templates/
    ├── base.html           # Base layout template
    ├── login.html          # Login page
    ├── register.html       # Registration page
    ├── dashboard.html      # User dashboard + prediction form
    ├── result.html         # Prediction result page
    └── admin.html          # Admin dashboard with charts
```

## 🛠️ Technology Stack

| Component       | Technology                         |
|-----------------|------------------------------------|
| Backend         | Python 3, Flask                    |
| Frontend        | HTML5, CSS3, Bootstrap 5           |
| ML Algorithm    | Naive Bayes (MultinomialNB)        |
| NLP Technique   | TF-IDF Vectorization               |
| Text Processing | NLTK (tokenize, stopwords, lemma)  |
| Database        | SQLite                             |
| Charts          | Chart.js                           |
| PDF Reports     | ReportLab                          |

## 🚀 Setup Instructions

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the ML Model

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Apply TF-IDF vectorization
- Train the Naive Bayes classifier
- Display accuracy, precision, recall, F1-score
- Save `model.pkl`, `vectorizer.pkl`, and `metrics.json`

### Step 3: Run the Web Application

```bash
python app.py
```

### Step 4: Open in Browser

Navigate to **http://localhost:5000**

- **Default Admin Login**: `admin` / `admin123`
- Register a new user account from the registration page

## ✨ Core Features

1. **User Authentication** — Registration and login with role-based access
2. **Job Posting Analysis** — Input title, description, company, location, salary
3. **NLP Text Preprocessing** — Lowercasing → Tokenization → Stopword Removal → Lemmatization
4. **TF-IDF Feature Extraction** — Converts text to numerical features
5. **Naive Bayes Prediction** — Classifies as REAL or FAKE
6. **Confidence Score** — Probability percentage for each class
7. **Prediction History** — All past analyses saved to SQLite
8. **Admin Dashboard** — Charts (pie, bar, line), user management, metrics display
9. **Model Metrics** — Accuracy, Precision, Recall, F1-Score shown on dashboard

## 🌟 Advanced Features

- 📧 **Email Alerts** — Automatic email when fake job detected (configure SMTP in `.env`)
- 📊 **Chart.js Dashboard** — Interactive doughnut, bar, and trend line charts
- 📄 **PDF Reports** — Download detailed prediction report as PDF
- 📁 **File Upload** — Upload `.txt` job postings for bulk analysis
- 🎯 **Confidence Display** — Visual probability bars for Real vs Fake

## 🤖 ML Pipeline Steps

1. **Load Dataset** — 200+ labeled job postings (CSV)
2. **Clean Data** — Handle nulls, combine text features
3. **Preprocess Text** — Lowercase, tokenize, remove stopwords, lemmatize
4. **Split Data** — 80% training / 20% testing (stratified)
5. **TF-IDF Vectorization** — Extract up to 5000 features with unigrams + bigrams
6. **Train Model** — MultinomialNB with alpha=0.1
7. **Evaluate** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## 📧 Email Configuration (Optional)

Edit `.env` file:
```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
```

## 📸 Screenshots Guide

| Page            | Description                                      |
|-----------------|--------------------------------------------------|
| Login           | Glassmorphism login card with gradient background |
| Register        | User registration with role selection             |
| Dashboard       | Prediction form, file upload, metrics, chart      |
| Result          | REAL/FAKE verdict with confidence bars            |
| Admin Dashboard | Stats cards, 3 charts, predictions table, users   |
| PDF Report      | Downloadable PDF with full prediction details     |

---

**Built with ❤️ using Python, Flask, Scikit-learn, and NLTK**
