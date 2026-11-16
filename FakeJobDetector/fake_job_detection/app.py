import os
import pickle
import sqlite3
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from preprocess import clean_text

app = Flask(__name__)
app.secret_key = 'fake-job-detection-secret-key'

# Load trained models
model = None
vectorizer = None

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("WARNING: Model files not found. Please run train_model.py first.")

# --- Database Implementation ---
def get_db_connection():
    conn = sqlite3.connect('app.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    ''')
    # Predictions table history
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT,
            company TEXT,
            salary TEXT,
            description TEXT,
            prediction_result TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    # Default admin user
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                 ('admin', generate_password_hash('admin123'), 'admin'))
    conn.commit()
    conn.close()

init_db()

# --- Routes ---
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))
            
        hashed_pw = generate_password_hash(password)
        
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.', 'danger')
        finally:
            conn.close()
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    if session.get('role') == 'admin':
        # Admin gets global view
        history = list(conn.execute('SELECT p.*, u.username FROM predictions p JOIN users u ON p.user_id = u.id ORDER BY p.timestamp DESC').fetchall())
        users = list(conn.execute('SELECT * FROM users').fetchall())
        conn.close()
        return render_template('dashboard.html', history=history, users=users)
    else:
        # Regular user view
        history = list(conn.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC', (session['user_id'],)).fetchall())
        conn.close()
        return render_template('dashboard.html', history=history)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        if not model or not vectorizer:
            flash("Model not loaded. Contact administrator.", "danger")
            return redirect(url_for('dashboard'))
            
        title = request.form.get('title', '')
        company = request.form.get('company', '')
        salary = request.form.get('salary', '')
        description = request.form.get('description', '')
        
        combined_text = f"{title} {company} {salary} {description}"
        processed_text = clean_text(combined_text)
        
        vec = vectorizer.transform([processed_text])
        prediction = model.predict(vec)[0]
        probabilities = model.predict_proba(vec)[0]
        
        # prediction: 0 -> real, 1 -> fake
        result_label = 'Fake' if prediction == 1 else 'Real'
        
        # calculate confidence based on the selected class
        confidence = probabilities[prediction] * 100
        
        # save to db
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO predictions (user_id, title, company, salary, description, prediction_result, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], title, company, salary, description, result_label, confidence))
        conn.commit()
        pred_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        conn.close()
        
        return redirect(url_for('result', pred_id=pred_id))
        
    return render_template('analyze.html')

@app.route('/result/<int:pred_id>')
def result(pred_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    pred = conn.execute('SELECT * FROM predictions WHERE id = ?', (pred_id,)).fetchone()
    conn.close()
    
    if not pred:
        flash("Result not found", "danger")
        return redirect(url_for('dashboard'))
        
    # Ensure regular users can only view their own predictions
    if session.get('role') != 'admin' and pred['user_id'] != session['user_id']:
        flash("Unauthorized access", "danger")
        return redirect(url_for('dashboard'))
        
    return render_template('result.html', pred=pred)

@app.route('/dataset_management', methods=['POST'])
def dataset_management():
    if session.get('role') != 'admin':
        return redirect(url_for('dashboard'))
        
    if 'dataset' in request.files:
        dataset = request.files['dataset']
        if dataset.filename != '':
            dataset.save('dataset.csv')
            flash('Dataset updated successfully!', 'success')
            
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=False, port=5000)
