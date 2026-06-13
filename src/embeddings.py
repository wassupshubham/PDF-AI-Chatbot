from langchain_huggingface import HuggingFaceEmbedembeddings, HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbedembeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )