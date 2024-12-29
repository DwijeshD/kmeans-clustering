import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # Tokenize text
    words = nltk.word_tokenize(text.lower())
    # Remove stop words
    filtered_words = [word for word in words if word.isalnum() and word not in stopwords.words('english')]
    return ' '.join(filtered_words)

def vectorize_text(processed_text):
    # Use TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    return vectorizer.fit_transform([processed_text]).toarray(), vectorizer