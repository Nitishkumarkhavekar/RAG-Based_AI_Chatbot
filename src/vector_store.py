import os
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

def create_vectorstore(
    chunks: List[Document], 
    embeddings_model: Embeddings
) -> FAISS:
    """
    Creates a FAISS vector store from a list of document chunks and an embeddings model.
    """
    if not chunks:
        logger.error("No chunks provided to create_vectorstore.")
        raise ValueError("Cannot create a vector store from an empty list of chunks.")
        
    try:
        logger.info(f"Generating FAISS vector store for {len(chunks)} chunks...")
        vectorstore = FAISS.from_documents(chunks, embeddings_model)
        logger.info("Successfully created FAISS vector store.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {str(e)}")
        raise RuntimeError(f"Failed to create FAISS vector store: {str(e)}")

def save_vectorstore(vectorstore: FAISS, folder_path: str, index_name: str = "index") -> None:
    """
    Saves the FAISS vector store locally to the specified folder path.
    """
    try:
        os.makedirs(folder_path, exist_ok=True)
        vectorstore.save_local(folder_path, index_name=index_name)
        logger.info(f"Vector store saved successfully to {os.path.join(folder_path, index_name)}")
    except Exception as e:
        logger.error(f"Error saving vector store to {folder_path}: {str(e)}")
        raise RuntimeError(f"Failed to save vector store: {str(e)}")

def load_vectorstore(
    folder_path: str, 
    embeddings_model: Embeddings, 
    index_name: str = "index"
) -> Optional[FAISS]:
    """
    Loads a FAISS vector store from a local folder.
    Returns None if the path or file does not exist.
    """
    index_file = os.path.join(folder_path, f"{index_name}.faiss")
    if not os.path.exists(index_file):
        logger.info(f"No existing FAISS index found at {index_file}.")
        return None
        
    try:
        logger.info(f"Loading FAISS vector store from {folder_path}...")
        # allow_dangerous_deserialization=True is required since langchain 0.1+ for loading local FAISS
        vectorstore = FAISS.load_local(
            folder_path, 
            embeddings_model, 
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        logger.info("Successfully loaded FAISS vector store.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading vector store from {folder_path}: {str(e)}")
        raise RuntimeError(f"Failed to load vector store: {str(e)}")

def similarity_search(
    vectorstore: FAISS, 
    query: str, 
    k: int = 4
) -> List[Document]:
    """
    Performs a similarity search in the vector store and returns top K documents.
    """
    if not vectorstore:
        logger.error("No vectorstore provided for similarity search.")
        return []
        
    try:
        logger.info(f"Performing similarity search for query: '{query}' (k={k})")
        results = vectorstore.similarity_search(query, k=k)
        logger.info(f"Search returned {len(results)} matches.")
        return results
    except Exception as e:
        logger.error(f"Error executing similarity search: {str(e)}")
        raise RuntimeError(f"Failed to search vector store: {str(e)}")
