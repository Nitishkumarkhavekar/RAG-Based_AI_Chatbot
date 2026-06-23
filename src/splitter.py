import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def split_documents(
    documents: List[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits a list of Documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of input LangChain Document objects.
        chunk_size: Maximum size of each chunk (default: 1000).
        chunk_overlap: Overlap between adjacent chunks (default: 200).
        
    Returns:
        List of split LangChain Document objects with updated metadata.
    """
    if not documents:
        logger.warning("Empty list of documents provided to split_documents.")
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    
    try:
        split_docs = splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks.")
        return split_docs
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        raise RuntimeError(f"Failed to split documents: {str(e)}")
