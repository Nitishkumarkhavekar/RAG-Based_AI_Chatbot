import os
import logging
try:
    from langchain_google_genai import GoogleGenAIEmbeddings
except ImportError:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings as GoogleGenAIEmbeddings

logger = logging.getLogger(__name__)

def get_embeddings_model(google_api_key: str = None) -> GoogleGenAIEmbeddings:
    """
    Initializes and returns the Google Gemini Embeddings model.
    
    Args:
        google_api_key: Optional Gemini API Key. If not provided, it will attempt
                        to load from the GOOGLE_API_KEY environment variable.
                        
    Returns:
        GoogleGenAIEmbeddings model instance.
    """
    api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        logger.error("Google API Key not found. Please set the GOOGLE_API_KEY in your environment or .env file.")
        raise ValueError("Google API Key is required to initialize Gemini Embeddings.")
        
    try:
        # gemini-embedding-001 is Google's recommended embedding model for general text tasks
        embeddings = GoogleGenAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        logger.info("Successfully initialized Gemini Embeddings (gemini-embedding-001)")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing Gemini Embeddings: {str(e)}")
        raise RuntimeError(f"Failed to initialize Gemini Embeddings: {str(e)}")
