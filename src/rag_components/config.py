import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

DEFAULT_CONFIG = {
    "chunk_size": 10000,
    "chunk_overlap": 500,
    "index_type": "Flat",
    "embedding_dimension": 768,
    "max_tokens": 2000,
    "log_level": "INFO",
    
    "retry_max_attempts": 5,
    "retry_base_delay": 1.0,
    "request_delay": 0.5,
}

def validate_config():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY environment variable is required")

