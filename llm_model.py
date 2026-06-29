import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

def get_llm(provider="gemini", model_name=None):
    if provider == "gemini":
        model = model_name or "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:
        model = model_name or "llama-3.1-8b-instant"
        return ChatGroq(
            model=model,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )