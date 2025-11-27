import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME =  os.getenv("GEMINI_MODEL_NAME")

if not GOOGLE_API_KEY:
    raise ValueError("Cannot find google model in .env")
