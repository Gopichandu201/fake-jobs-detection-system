import os
import re
import json
import joblib
import sqlite3
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from io import BytesIO
from functools import wraps
import numpy as np

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_file, jsonify
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data silently
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ── Flask App Setup ───────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fakejobdetector_secret_2024")
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB max upload

# ── Load ML Model & Metrics ──────────────────────────────────────────────────
try:
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
except FileNotFoundError:
    model = None
    vectorizer = None

metrics = {}
if os.path.exists("metrics.json"):
    with open("metrics.json", "r") as f:
        metrics = json.load(f)

# ── NLP Preprocessing ────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def extract_suspicious_keywords(text, vectorizer, model, top_n=5):
    """
    Extracts the top contributing keywords for the 'Fake' class from the input text.
    """
    if not model or not vectorizer:
        return []

    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    
    # Get the log probabilities for the 'fake' class (index 1)
    # MultinomialNB stores log probabilities of features given a class in `feature_log_prob_`
    fake_log_probs = model.feature_log_prob_[1]
    
    # Find which features in the vocabulary are present in the text
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the indices of non-zero features in the transformed vector
    nonzero_indices = vec.nonzero()[1]
    
    # Map feature indices to their corresponding log probability and name
    word_scores = []
    for idx in nonzero_indices:
        word_scores.append((feature_names[idx], fake_log_probs[idx]))
        
    # Sort by log probability (highest contribution to fake class)
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N keywords
    return [word for word, score in word_scores[:top_n]]

# ── Database Setup ────────────────────────────────────────────────────────────
DB_NAME = "database.db"

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT DEFAULT '',
        role TEXT DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Check if predictions table exists and has risk_percentage column
    c.execute("PRAGMA table_info(predictions)")
    columns = [row[1] for row in c.fetchall()]
    
    if not columns:
        c.execute('''CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            title TEXT,
            description TEXT,
            company TEXT DEFAULT '',
            location TEXT DEFAULT '',
            salary TpEXT DEFAULT '',
            result TEXT,
            confidence REAL DEFAULT 0,
            prob_real REAL DEFAULT 0,
            prob_fake REAL DEFAULT 0,
            risk_percentage REAL DEFAULT 0,
            suspicious_keywords TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
    else:
        # Migrate existing table if needed
        if 'risk_percentage' not in columns:
            c.execute("ALTER TABLE predictions ADD COLUMN risk_percentage REAL DEFAULT 0")
        if 'suspicious_keywords' not in columns:
            c.execute("ALTER TABLE predictions ADD COLUMN suspicious_keywords TEXT DEFAULT ''")

    # Seed admin user if not exists
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)",
                  ("admin", "admin123", "admin@fakejobdetector.com", "admin"))
    conn.commit()
    conn.close()

init_db()

# ── Auth Decorator ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            flash("Please login to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session or session.get("role") != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ── Email Alert ───────────────────────────────────────────────────────────────
def send_fake_alert_email(user_email, title, risk_percentage):
    try:
        smtp_host = os.environ.get("SMTP_HOST", "")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_pass = os.environ.get("SMTP_PASS", "")
        if not smtp_host or not smtp_user: return

        msg = MIMEText(
            f"⚠️ FAKE JOB ALERT\n\n"
            f"The job posting titled \"{title}\" has been detected as FAKE "
            f"with a {risk_percentage:.1f}% risk score.\n\n"
            f"Please exercise caution.\n\n"
            f"— Fake Job Detector System"
        )
        msg["Subject"] = f"⚠️ Fake Job Alert: {title}"
        msg["From"] = smtp_user
        msg["To"] = user_email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
    except Exception:
        pass

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "username" in session:
        if session.get("role") == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password)).fetchone()
        conn.close()
        if user:
            session["username"] = user["username"]
            session["role"] = user["role"]
            session["email"] = user["email"]
            flash(f"Welcome back, {username}!", "success")
            if user["role"] == "admin": return redirect(url_for("admin_dashboard"))
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        email = request.form.get("email", "").strip()
        role = request.form.get("role", "user")

        if not username or not password:
            flash("Username and password required.", "warning")
            return render_template("register.html")

        conn = get_db()
        if conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone():
            conn.close()
            flash("Username already exists.", "danger")
            return render_template("register.html")

        conn.execute("INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)",
                     (username, password, email, role))
        conn.commit()
        conn.close()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    predictions = conn.execute("SELECT * FROM predictions WHERE username = ? ORDER BY created_at DESC", (session["username"],)).fetchall()
    conn.close()
    return render_template("dashboard.html", predictions=predictions, metrics=metrics)

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if not model or not vectorizer:
        flash("Model not loaded! Please run `python train_model.py` first.", "danger")
        return redirect(url_for("dashboard"))

    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    company = request.form.get("company", "").strip()
    location = request.form.get("location", "").strip()
    salary = request.form.get("salary", "").strip()

    if not title and not description:
        flash("Please provide at least a job title or description.", "warning")
        return redirect(url_for("dashboard"))

    raw_text = f"{title} {company} {location} {salary} {description}"
    processed = preprocess_text(raw_text)

    # Predict
    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]

    result = "FAKE" if prediction == 1 else "REAL"
    prob_real = round(probabilities[0] * 100, 2)
    prob_fake = round(probabilities[1] * 100, 2)
    
    # Calculate Risk Percentage and Keywords
    risk_percentage = prob_fake
    confidence = prob_fake if prediction == 1 else prob_real
    
    keywords = extract_suspicious_keywords(raw_text, vectorizer, model, top_n=5)
    keywords_str = ",".join(keywords) if result == "FAKE" else ""

    conn = get_db()
    conn.execute(
        """INSERT INTO predictions
           (username, title, description, company, location, salary, result, confidence, prob_real, prob_fake, risk_percentage, suspicious_keywords)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session["username"], title, description, company, location, salary,
         result, confidence, prob_real, prob_fake, risk_percentage, keywords_str)
    )
    conn.commit()
    pred_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    if result == "FAKE" and session.get("email"):
        send_fake_alert_email(session["email"], title, risk_percentage)

    return render_template("result.html",
                           result=result,
                           confidence=confidence,
                           prob_real=prob_real,
                           prob_fake=prob_fake,
                           risk_percentage=risk_percentage,
                           keywords=keywords if result == "FAKE" else [],
                           title=title,
                           description=description,
                           company=company,
                           location=location,
                           salary=salary,
                           pred_id=pred_id)

@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    if not model or not vectorizer:
        flash("Model not loaded!", "danger")
        return redirect(url_for("dashboard"))

    if "file" not in request.files:
        flash("No file uploaded.", "warning")
        return redirect(url_for("dashboard"))

    file = request.files["file"]
    if not file.filename.endswith(".txt"):
        flash("Only .txt files are supported.", "danger")
        return redirect(url_for("dashboard"))

    content = file.read().decode("utf-8", errors="ignore").strip()
    if not content:
        flash("File is empty.", "warning")
        return redirect(url_for("dashboard"))

    processed = preprocess_text(content)
    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]

    result = "FAKE" if prediction == 1 else "REAL"
    prob_real = round(probabilities[0] * 100, 2)
    prob_fake = round(probabilities[1] * 100, 2)
    risk_percentage = prob_fake
    confidence = prob_fake if prediction == 1 else prob_real
    
    keywords = extract_suspicious_keywords(content, vectorizer, model, top_n=5)
    keywords_str = ",".join(keywords) if result == "FAKE" else ""
    title = file.filename.replace(".txt", "")
    description = content[:1000]

    conn = get_db()
    conn.execute(
        """INSERT INTO predictions
           (username, title, description, result, confidence, prob_real, prob_fake, risk_percentage, suspicious_keywords)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session["username"], title, description, result, confidence, prob_real, prob_fake, risk_percentage, keywords_str)
    )
    conn.commit()
    pred_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    if result == "FAKE" and session.get("email"):
        send_fake_alert_email(session["email"], title, risk_percentage)

    return render_template("result.html",
                           result=result, confidence=confidence, prob_real=prob_real, prob_fake=prob_fake,
                           risk_percentage=risk_percentage, keywords=keywords if result == "FAKE" else [],
                           title=title, description=description, company="", location="", salary="", pred_id=pred_id)

# ── Chatbot System ────────────────────────────────────────────────────────────

@app.route("/chatbot/<int:pred_id>")
@login_required
def chatbot_page(pred_id):
    conn = get_db()
    pred = conn.execute("SELECT * FROM predictions WHERE id = ? AND username = ?", (pred_id, session["username"])).fetchone()
    conn.close()
    if not pred:
        flash("Prediction not found or unauthorized.", "danger")
        return redirect(url_for("dashboard"))
    
    return render_template("chatbot.html", pred=pred)

@app.route("/api/chat/<int:pred_id>", methods=["POST"])
@login_required
def api_chat(pred_id):
    question = request.json.get("question", "").strip().lower()
    if not question:
        return jsonify({"answer": "I didn't catch that. Please ask a valid question."})

    conn = get_db()
    pred = conn.execute("SELECT * FROM predictions WHERE id = ? AND username = ?", (pred_id, session["username"])).fetchone()
    conn.close()

    if not pred:
        return jsonify({"answer": "Unauthorized or unknown prediction reference."})

    # Rule-Based Chatbot Logic
    answer = ""
    is_fake = pred['result'] == 'FAKE'
    
    if "why" in question and "fake" in question:
        if is_fake:
            kws = pred['suspicious_keywords'].split(',') if pred['suspicious_keywords'] else []
            kw_str = "<ul>" + "".join([f"<li><strong>{k.strip()}</strong></li>" for k in kws]) + "</ul>"
            answer = f"The job posting was flagged as <strong>FAKE</strong> because my AI model analyzed the text and found patterns commonly associated with scams. Specifically, the following suspicious keywords contributed significantly to the fake classification: {kw_str} <br>Additionally, the overall tone and structure matched known fraudulent postings with a {pred['risk_percentage']}% risk score."
        else:
            answer = "Actually, this job was evaluated as <strong>REAL</strong>. The patterns used in the description are consistent with authentic job postings."
            
    elif "avoid" in question or "scam" in question:
        answer = """Here are the top 5 ways to avoid job scams:
        <ol>
            <li><strong>Unrealistic Salaries:</strong> Be wary of postings offering high pay for little work or 'no experience needed'.</li>
            <li><strong>Upfront Fees:</strong> Legitimate companies never ask you to pay for equipment, training, or background checks.</li>
            <li><strong>Vague Job Descriptions:</strong> Scammers often use generic terms that appeal to a wide audience without detailing actual duties.</li>
            <li><strong>Suspicious Emails:</strong> Check if emails come from free services (like @gmail.com) instead of a corporate domain.</li>
            <li><strong>Instant Hires:</strong> If you are offered a job on the spot through a chat app without an interview, it's highly likely a scam.</li>
        </ol>"""

    elif "confidence" in question or "score" in question or "risk" in question:
        answer = f"The <strong>Confidence Score</strong> represents how certain the AI is about its prediction. For this posting, the model calculated a <strong>{pred['confidence']}% probability</strong> that it belongs to the <code>{pred['result']}</code> class.<br><br>The model uses TF-IDF vectorization to convert text into numbers and a Naive Bayes algorithm to compute these probabilities based on historical data."

    elif "keyword" in question or "suspicious words" in question:
        if is_fake:
            kws = pred['suspicious_keywords'].split(',') if pred['suspicious_keywords'] else []
            answer = f"I detected the following suspicious keywords in the job description: <strong>{', '.join(kws)}</strong>. These words frequently appear in fake job datasets."
        else:
            answer = "Since this job was classified as REAL, I didn't flag any keywords as significantly suspicious."

    else:
        answer = "I'm a rule-based explainability chatbot. You can ask me questions like: <br>• 'Why is this job fake?'<br>• 'How to avoid job scams?'<br>• 'What is confidence score?'<br>• 'What were the suspicious keywords?'"

    return jsonify({"answer": answer})

# ── Download PDF Report ───────────────────────────────────────────────────────
@app.route("/download_report/<int:pred_id>")
@login_required
def download_report(pred_id):
    conn = get_db()
    pred = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    conn.close()

    if not pred: return redirect(url_for("dashboard"))

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch, leftMargin=0.75*inch, rightMargin=0.75*inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=20, textColor=colors.HexColor('#1a1a2e'), spaceAfter=20)
    story.append(Paragraph("Fake Job Detection Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#6c63ff')))
    story.append(Spacer(1, 20))

    result_color = colors.HexColor('#e74c3c') if pred['result'] == 'FAKE' else colors.HexColor('#27ae60')
    result_style = ParagraphStyle('Result', parent=styles['Heading1'], fontSize=28, textColor=result_color, alignment=1)
    story.append(Paragraph(f"Verdict: {pred['result']}", result_style))
    story.append(Spacer(1, 5))

    conf_style = ParagraphStyle('Confidence', parent=styles['Normal'], fontSize=14, alignment=1, textColor=colors.gray)
    story.append(Paragraph(f"Confidence: {pred['confidence']:.1f}% | Risk: {pred['risk_percentage']:.1f}%", conf_style))
    story.append(Spacer(1, 25))

    kws = pred['suspicious_keywords'].replace(',', ', ') if pred['suspicious_keywords'] else "None"
    
    data = [
        ["Field", "Value"],
        ["Job Title", pred['title'] or "N/A"],
        ["Company", pred['company'] or "N/A"],
        ["Location", pred['location'] or "N/A"],
        ["Description", Paragraph(pred['description'][:300] or "N/A", styles['Normal'])],
        ["Probability (Real)", f"{pred['prob_real']}%"],
        ["Probability (Fake)", f"{pred['prob_fake']}%"],
        ["Risk Percentage", f"{pred['risk_percentage']}%"],
        ["Suspicious Words", kws],
        ["Analyzed By", pred['username']],
    ]

    table = Table(data, colWidths=[1.8 * inch, 4.5 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6c63ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f0f0f5')),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ddd')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(table)
    story.append(Spacer(1, 30))

    disc_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=9, textColor=colors.gray, alignment=1)
    story.append(Paragraph("This explainable AI report uses joblib, TF-IDF, and Naive Bayes for detection.", disc_style))

    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"Report_{pred_id}_{pred['result']}.pdf", mimetype='application/pdf')

# ── Admin Dashboard ───────────────────────────────────────────────────────────
@app.route("/admin")
@admin_required
def admin_dashboard():
    conn = get_db()
    predictions = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC").fetchall()
    users = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    conn.close()

    total_predictions = len(predictions)
    total_real = sum(1 for p in predictions if p['result'] == 'REAL')
    total_fake = sum(1 for p in predictions if p['result'] == 'FAKE')
    
    return render_template("admin.html", predictions=predictions, users=users,
                           total_predictions=total_predictions, total_real=total_real,
                           total_fake=total_fake, total_users=len(users), metrics=metrics)

@app.route("/api/stats")
@login_required
def api_stats():
    conn = get_db()
    if session.get("role") == "admin":
        predictions = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC").fetchall()
    else:
        predictions = conn.execute("SELECT * FROM predictions WHERE username = ? ORDER BY created_at DESC", (session["username"],)).fetchall()
    
    total_real = sum(1 for p in predictions if p['result'] == 'REAL')
    total_fake = sum(1 for p in predictions if p['result'] == 'FAKE')

    daily = {}
    for p in predictions:
        if p['created_at']:
            day = p['created_at'][:10]
            if day not in daily: daily[day] = {"real": 0, "fake": 0}
            if p['result'] == 'REAL': daily[day]["real"] += 1
            else: daily[day]["fake"] += 1

    sorted_days = sorted(daily.keys())[-30:]
    conn.close()
    return jsonify({
        "total_real": total_real, "total_fake": total_fake,
        "daily_labels": sorted_days,
        "daily_real": [daily[d]["real"] for d in sorted_days],
        "daily_fake": [daily[d]["fake"] for d in sorted_days],
    })

@app.route("/admin/delete/<int:pred_id>")
@admin_required
def delete_prediction(pred_id):
    conn = get_db()
    conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    conn.commit()
    conn.close()
    flash("Prediction deleted.", "info")
    return redirect(url_for("admin_dashboard"))

# ── Run Server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/img", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, port=5000)