import pickle
from preprocess import clean_text

def load_models():
    """Loads the trained model and vectorizer from disk."""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        print("Error: model.pkl or vectorizer.pkl not found. Please run train_model.py first.")
        return None, None

def predict_job(title, company, salary, description):
    """
    Predicts whether a job posting is real or fake.
    Returns:
        prediction_label (str): "Real" or "Fake"
        confidence (float): Probability score in percentage
    """
    model, vectorizer = load_models()
    if not model or not vectorizer:
        return None, None

    # Combine text the same way it was done during training
    combined_text = f"{title} {company} {salary} {description}"
    
    # Preprocess the text
    processed_text = clean_text(combined_text)
    
    # Vectorize the text
    vec = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    
    # Post-process results (0 -> real, 1 -> fake)
    result_label = 'Fake' if prediction == 1 else 'Real'
    confidence = probabilities[prediction] * 100
    
    return result_label, confidence

if __name__ == "__main__":
    print("--- Fake Job Detection Prediction Test ---")
    
    # Example 1: Real Job
    title1 = "Software Engineer"
    company1 = "Google"
    salary1 = "120000"
    desc1 = "Design and develop scalable web applications using Python and JavaScript frameworks. Collaborate with cross-functional teams to define and implement new features. Write clean maintainable code following best practices."
    
    print("\nTest Job 1 (Expected: Real):")
    print(f"Title: {title1}")
    res1, conf1 = predict_job(title1, company1, salary1, desc1)
    if res1:
        print(f"Result: {res1} (Confidence: {conf1:.2f}%)")

    # Example 2: Fake Job
    title2 = "Easy Money Online"
    company2 = "Unknown Company"
    salary2 = "300000"
    desc2 = "Make money from home with just your phone. No experience needed no interview required. Earn up to 5000 dollars per day just by clicking ads and sharing links."
    
    print("\nTest Job 2 (Expected: Fake):")
    print(f"Title: {title2}")
    res2, conf2 = predict_job(title2, company2, salary2, desc2)
    if res2:
        print(f"Result: {res2} (Confidence: {conf2:.2f}%)")
