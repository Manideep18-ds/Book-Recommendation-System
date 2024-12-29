import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# The main recommendation fucntion which is getting used in app.py
def recommend_books(book_title, embedding_dict, bert_model):
    """
    Recommend books based on a given book title.
    - book_title: str, title of the input book.
    - embedding_dict: dict, mapping of book names to their embeddings.
    - bert_model: SentenceTransformer, preloaded BERT model.
    - Returns: list, recommended book titles.
    """
    
    # Get embedding for the input book
    input_embedding = bert_model.encode([book_title])

    # Compute cosine similarity with all embeddings
    book_names = list(embedding_dict.keys())
    all_embeddings = np.array(list(embedding_dict.values()))
    cosine_sim = cosine_similarity(input_embedding, all_embeddings).flatten()

    # Get top 5 recommendations
    similar_indices = cosine_sim.argsort()[-6:-1][::-1]  # Exclude the input book itself
    recommendations = [book_names[i] for i in similar_indices]
    
    return recommendations

# 1. Count Vectorizer
def compute_count_vectorizer(data, column='content'):
    """
    Compute the Count Vectorizer representation for the dataset.
    - data: DataFrame, preprocessed dataset.
    - column: str, name of the column to vectorize.
    - Returns: tuple, (Count Vectorizer matrix, fitted CountVectorizer object).
    """
    vectorizer = CountVectorizer(stop_words='english')
    count_matrix = vectorizer.fit_transform(data[column])
    return count_matrix, vectorizer

# 2. TF-IDF Vectorizer
def compute_tfidf_vectorizer(data, column='content'):
    """
    Compute the TF-IDF representation for the dataset.
    - data: DataFrame, preprocessed dataset.
    - column: str, name of the column to vectorize.
    - Returns: tuple, (TF-IDF matrix, fitted TfidfVectorizer object).
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data[column])
    return tfidf_matrix, vectorizer

# 3. BERT Embeddings
def compute_bert_embeddings(data, column='content', model_name='all-MiniLM-L6-v2'):
    """
    Compute BERT embeddings for the dataset using SentenceTransformer.
    - data: DataFrame, preprocessed dataset.
    - column: str, name of the column to encode.
    - model_name: str, name of the SentenceTransformer model.
    - Returns: numpy.ndarray, BERT embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(data[column].tolist(), show_progress_bar=True)
    return np.array(embeddings), model
