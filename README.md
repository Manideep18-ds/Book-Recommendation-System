# Book Recommendation System

This is a content-based book recommendation system implemented in Python.

## Files
- `app.py`: Main script to run the Gradio app.
- `recommender.py`: Refactored fucntion code which contains preprocessing, embedding, recommender fucntions.
- `books_summary.csv`: Dataset with book details.
- `requirements.txt`: Dependencies.
- `eda_recommender.ipynb`- The actual file where eda, embeddings,vectorizers(experimented with tfidf and count vectorizer also) , by using cosine similarity tested few input textbook names and saved bert embeddings and pickle to reduce latency

## How to Run
1. Install dependencies:
pip install -r requirements.txt

2. Run the app:

3. Access the Gradio UI in your browser.

## Deployment
To deploy on Hugging Face Spaces, include all files and ensure the dataset is accessible.


