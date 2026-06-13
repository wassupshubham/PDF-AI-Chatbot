from langchain_chroma import Chroma

def create_vectorstore(chunks, embeddings):
    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )