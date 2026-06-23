import os
import shutil
import logging
import google.generativeai as genai
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def validate_api_key(google_api_key: str) -> bool:
    """
    Validates a Google Gemini API Key by attempting a lightweight API call (listing models).
    """
    if not google_api_key:
        return False
        
    try:
        genai.configure(api_key=google_api_key)
        # Attempt to list models to verify the key
        models = genai.list_models()
        # If no error was raised, the API key is valid
        return True
    except Exception as e:
        logger.warning(f"API key validation failed: {str(e)}")
        return False

def get_file_size_formatted(num_bytes: int) -> str:
    """
    Formats raw bytes size into human-readable string (KB, MB).
    """
    for unit in ['Bytes', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"

def save_uploaded_file(uploaded_file, target_directory: str) -> str:
    """
    Saves a Streamlit UploadedFile to a local target directory and returns the path.
    """
    os.makedirs(target_directory, exist_ok=True)
    file_path = os.path.join(target_directory, uploaded_file.name)
    
    # Reset stream pointer
    uploaded_file.seek(0)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
        
    logger.info(f"Saved uploaded file: {uploaded_file.name} to {file_path}")
    return file_path

def clean_directory(directory_path: str) -> None:
    """
    Safely deletes all files in a directory.
    """
    if not os.path.exists(directory_path):
        return
        
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f"Failed to delete {file_path}. Reason: {e}")
