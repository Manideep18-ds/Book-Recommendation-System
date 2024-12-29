from recommender import  recommend_books
import gradio as gr
import pickle

def gradio_interface(book_title):
    """
    Gradio wrapper for book recommendation.
    - book_title: str, title of the book entered by the user.
    - Returns: str, recommended book titles separated by newlines.
    """
    print("Loading precomputed embeddings and model of book_summaries")
    
    with open("book_embeddings.pkl", "rb") as file:
        embedding_dict = pickle.load(file)
    with open("bert_model.pkl", "rb") as file:
        bert_model = pickle.load(file)
    print("Loading completed!!")
    recommendations=recommend_books(book_title, embedding_dict, bert_model)
    
    return "\n".join(recommendations)

# Define Gradio UI
ui = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter a Book Title"),
    outputs=gr.Textbox(label="Recommended Books"),
    title="Book Recommendation System"
)

if __name__ == "__main__":
    ui.launch()
