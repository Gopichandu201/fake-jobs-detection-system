# AI-Driven Fake Job Posting Detection System Using Machine Learning and NLP

## 📌 Project Overview

The **AI-Driven Fake Job Posting Detection System** is a machine learning-based web application designed to identify fraudulent job postings and distinguish them from genuine job opportunities.

The system uses Natural Language Processing (NLP), TF-IDF (Term Frequency–Inverse Document Frequency), and the Naive Bayes classification algorithm to analyze job descriptions and predict whether a job posting is **Real or Fake**.

The main objective is to help job seekers identify potentially fraudulent recruitment advertisements and reduce the risk of online job scams.

## 🎯 Objectives

- Identify fake and genuine job postings using machine learning.
- Analyze job descriptions using NLP techniques.
- Extract meaningful text features using TF-IDF.
- Classify job postings using the Naive Bayes algorithm.
- Provide a simple and user-friendly web interface.
- Help job seekers make safer employment decisions.

## 🚀 Key Features

- **User Authentication:** Secure login and registration.
- **Job Description Analysis:** Analyze job posting content.
- **Fake Job Detection:** Classify job postings as Real or Fake.
- **Machine Learning Model:** Naive Bayes classification.
- **Text Processing:** NLP-based text preprocessing and TF-IDF feature extraction.
- **Admin Dashboard:** Manage and monitor application activities.
- **Model Evaluation:** Evaluate performance using accuracy, precision, recall, F1-score, and confusion matrix.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Naive Bayes |
| NLP Technique | TF-IDF |
| Libraries | Pandas, NumPy, Scikit-learn |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Database | As implemented |

## ⚙️ System Workflow

1. User logs into the application.
2. User enters or submits a job description.
3. The system preprocesses the submitted text.
4. TF-IDF converts text into numerical features.
5. The trained Naive Bayes model analyzes the features.
6. The system predicts whether the job posting is Real or Fake.
7. The prediction result is displayed to the user.

## 📂 Project Structure

```text
Fake-Job-Detection/
│
├── app.py
├── model/
│   └── fake_job_model.pkl
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   └── admin_dashboard.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── dataset/
│   └── fake_job_postings.csv
│
├── requirements.txt
└── README.md
```

*Note: Update the folder structure according to your actual project files.*

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd Fake-Job-Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app.py
```

### 7. Open in Browser

```text
http://127.0.0.1:5000/
```

## 📊 Model Evaluation

The machine learning model can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These evaluation metrics help measure the model's ability to identify fraudulent and genuine job postings.

## 🔮 Future Enhancements

- Integrate advanced machine learning and deep learning algorithms.
- Improve prediction accuracy using larger datasets.
- Add real-time job posting analysis.
- Develop an explainable AI feature to highlight suspicious job descriptions.
- Deploy the application on a cloud platform.

## 👨‍💻 Developer

**K Gopi Chandu**

Computer Science and Engineering Graduate

**Areas of Interest:** Python Development | Machine Learning | Data Analytics | Artificial Intelligence

## 📜 License

This project is developed for academic and educational purposes.
