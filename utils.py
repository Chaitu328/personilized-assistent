import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources if they're not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    """Clean and normalize text"""
    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single one
    text = re.sub(r'\s+', ' ', text)
    # Remove any unusual characters
    text = re.sub(r'[^\w\s\.\,\?\!\:\;\-\'\"]', '', text)
    return text.strip()

def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks for processing"""
    words = word_tokenize(text)
    chunks = []
    
    i = 0
    while i < len(words):
        # Get chunk with specified size
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        # Move forward with overlap
        i += chunk_size - overlap
    
    return chunks

def extract_topics(text):
    """Extract potential topics from the text"""
    # This is a simplified topic extraction
    # In a real-world scenario, we might use more sophisticated approaches like LDA
    
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    
    # Filter out stop words and short words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count word frequency
    word_freq = {}
    for word in filtered_words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
    # Get the most frequent words as potential topics
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return the top words as potential topics
    topics = [word for word, _ in sorted_words[:10]]
    
    return topics

def chunk_for_embeddings(text, chunk_size=500):
    """Split text into chunks appropriate for embedding"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size <= chunk_size:
            current_chunk.append(sentence)
            current_size += sentence_size
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
