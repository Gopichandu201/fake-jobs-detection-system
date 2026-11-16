import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data quietly if not present
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

def clean_text(text):
    """
    Cleans the input text by applying:
    1. Lowercase conversion
    2. Remove punctuation
    3. Tokenization
    4. Stopword removal
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercase conversion
    text = text.lower()
    
    # 2. Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Stopword removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    return " ".join(filtered_tokens)
