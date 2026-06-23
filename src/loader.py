import os
import io
import logging
from typing import List, Union
from langchain_core.documents import Document
import pypdf
import docx2txt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdf(file_source: Union[str, io.BytesIO], filename: str) -> List[Document]:
    """
    Extracts text from a PDF file and returns a list of LangChain Document objects.
    Supports both file paths and byte streams.
    """
    documents = []
    try:
        if isinstance(file_source, str):
            # It's a file path
            with open(file_source, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={"source": filename, "page": i + 1}
                            )
                        )
        else:
            # It's a byte stream (Streamlit UploadedFile)
            pdf_reader = pypdf.PdfReader(file_source)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": filename, "page": i + 1}
                        )
                    )
        logger.info(f"Successfully loaded PDF: {filename} ({len(documents)} pages)")
    except Exception as e:
        logger.error(f"Error loading PDF {filename}: {str(e)}")
        raise RuntimeError(f"Failed to process PDF {filename}: {str(e)}")
    
    return documents

def load_txt(file_source: Union[str, io.BytesIO], filename: str) -> List[Document]:
    """
    Extracts text from a TXT file and returns a list of LangChain Document objects.
    Supports both file paths and byte streams with encoding fallbacks.
    """
    documents = []
    try:
        content = ""
        if isinstance(file_source, str):
            # It's a file path
            for encoding in ("utf-8", "latin-1", "cp1252"):
                try:
                    with open(file_source, "r", encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
        else:
            # It's a byte stream (Streamlit UploadedFile)
            file_bytes = file_source.read()
            # Reset stream pointer just in case
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            
            for encoding in ("utf-8", "latin-1", "cp1252"):
                try:
                    content = file_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
        
        if not content:
            raise ValueError("Empty or unreadable text content")
            
        documents.append(
            Document(
                page_content=content,
                metadata={"source": filename}
            )
        )
        logger.info(f"Successfully loaded TXT: {filename}")
    except Exception as e:
        logger.error(f"Error loading TXT {filename}: {str(e)}")
        raise RuntimeError(f"Failed to process TXT {filename}: {str(e)}")
        
    return documents

def load_docx(file_source: Union[str, io.BytesIO], filename: str) -> List[Document]:
    """
    Extracts text from a DOCX file and returns a list of LangChain Document objects.
    Supports both file paths and byte streams.
    """
    documents = []
    try:
        # docx2txt accepts either a file path or a file-like object
        text = docx2txt.process(file_source)
        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": filename}
                )
            )
        logger.info(f"Successfully loaded DOCX: {filename}")
    except Exception as e:
        logger.error(f"Error loading DOCX {filename}: {str(e)}")
        raise RuntimeError(f"Failed to process DOCX {filename}: {str(e)}")
        
    return documents

def load_document(file_source: Union[str, io.BytesIO], filename: str) -> List[Document]:
    """
    Routes document loading to the appropriate function based on the file extension.
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == ".pdf":
        return load_pdf(file_source, filename)
    elif ext == ".txt":
        return load_txt(file_source, filename)
    elif ext in (".docx", ".doc"):
        return load_docx(file_source, filename)
    else:
        logger.warning(f"Unsupported file format: {ext} for file {filename}")
        raise ValueError(f"Unsupported file format '{ext}'. Only PDF, TXT, and DOCX are supported.")

def load_multiple_documents(uploaded_files: List) -> List[Document]:
    """
    Processes a list of uploaded files or file paths and returns aggregated Documents.
    """
    all_documents = []
    for u_file in uploaded_files:
        # Check if u_file is a file path or a Streamlit UploadedFile
        if isinstance(u_file, str):
            filename = os.path.basename(u_file)
            file_source = u_file
        else:
            filename = u_file.name
            file_source = u_file
            
        try:
            docs = load_document(file_source, filename)
            all_documents.extend(docs)
        except Exception as e:
            logger.error(f"Failed to load {filename}: {str(e)}")
            # Propagate error so frontend/user can see it
            raise e
            
    return all_documents
