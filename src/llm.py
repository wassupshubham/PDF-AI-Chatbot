import os
from langchain_groq import ChatGroq

def get_llm():
    # SECURE: Pulls key from Render/Local system variables automatically
    api_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key
    )