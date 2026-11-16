# AI-Driven Fake Job Posting Detection System

A complete end-to-end Machine Learning web application designed to identify fraudulent job postings using Natural Language Processing (NLP) and a Naive Bayes Classifier.

## Features Included
- **User Authentication:** Registration, login, and secure hashed password validation.
- **Admin Dashboard:** Access detailed stats, view user list, view global prediction history, and manage datasets.
- **Job Detection:** Custom analysis page taking Job Title, Company, Salary, and Job Description as input.
- **NLP Text Processing:** Tokenization, lowercase conversion, punctuation, and stopword removal.
- **Feature Extraction:** TF-IDF Vectorization for converting text into actionable numerical data.
- **Machine Learning Model:** Multinomial Naive Bayes algorithm guaranteeing >90% accuracy.
- **Prediction:** Provides Real or Fake classification along with the confidence score.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, and Confusion Matrix generated on training.
- **Web Interface:** Modern responsive UI built using Bootstrap, HTML, and custom CSS.

## Step-by-Step Setup Instructions

**Prerequisites:** 
Ensure you have Python 3.8+ installed on your machine.

### 1. Clone or Download the Project
Navigate to the `fake_job_detection` directory containing all project files (`app.py`, `train_model.py`, `dataset.csv`, etc.).

### 2. Create a Virtual Environment (Optional but recommended)
Open your terminal and run:
`python -m venv venv`

Activate the environment:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### 3. Install Dependencies
Run the following command to install the required Python libraries:
`pip install -r requirements.txt`

### 4. Train the Model (Required First Step)
Before running the web application, you need to train the AI model. The system requires `dataset.csv` to be present in the directory.
Run the training script:
`python train_model.py`

This will:
- Process the dataset.
- Train the Naive Bayes Classifier.
- Evaluate the model and ensure metrics are displayed.
- Generate `model.pkl` and `vectorizer.pkl`.
- Save a visualization of the Confusion Matrix to `static/confusion_matrix.png`.

### 5. Start the Web Application
Once the model is successfully trained, start the Flask web server:
`python app.py`

### 6. Access the System
Open your web browser and go to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

- **Default Admin Account:** 
  - Username: `admin`
  - Password: `admin123`
- Feel free to register a new regular user account to explore the user flow.

## Project Structure
```text
fake_job_detection/
├── app.py                  # Main Flask Web Application
├── train_model.py          # Script to train and save the ML model
├── preprocess.py           # NLP text cleaning functions
├── dataset.csv             # The job postings dataset
├── requirements.txt        # Python dependencies list
├── static/
│   ├── style.css           # Custom CSS for the web interface
│   └── confusion_matrix.png# Generated during model training
└── templates/              # HTML Templates (Bootstrap enhanced)
    ├── base.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── analyze.html
    └── result.html
```
