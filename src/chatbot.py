import os
import logging
from typing import List, Dict, Tuple, Any
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

def get_llm(google_api_key: str = None, temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """
    Initializes and returns the ChatGoogleGenerativeAI LLM instance.
    """
    api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Google API Key not found for chatbot initialization.")
        raise ValueError("Google API Key is required to initialize the Chatbot.")
        
    try:
        # Using gemini-2.5-flash as the default fast and context-rich model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=2048
        )
        logger.info("Successfully initialized ChatGoogleGenerativeAI (gemini-1.5-flash)")
        return llm
    except Exception as e:
        logger.error(f"Error initializing Gemini LLM: {str(e)}")
        raise RuntimeError(f"Failed to initialize Gemini LLM: {str(e)}")

def format_chat_history(chat_history: List[Dict[str, str]]) -> List[Any]:
    """
    Converts list of user/assistant message dicts to LangChain Message objects.
    """
    messages = []
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages

def format_context(context_docs: List[Document]) -> str:
    """
    Formats list of retrieved Documents into a single string context with source markers.
    """
    formatted_chunks = []
    for i, doc in enumerate(context_docs):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", None)
        page_str = f" (Page {page})" if page else ""
        formatted_chunks.append(
            f"--- Document Chunk {i+1} | Source: {source}{page_str} ---\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(formatted_chunks)

def generate_answer(
    query: str,
    context_docs: List[Document],
    chat_history: List[Dict[str, str]],
    google_api_key: str = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generates a RAG-based answer using retrieved context documents and chat history.
    
    Returns:
        A tuple: (answer_string, list_of_source_metadata)
    """
    llm = get_llm(google_api_key=google_api_key)
    
    formatted_context = format_context(context_docs)
    lc_history = format_chat_history(chat_history)
    
    # Define RAG prompt template
    system_instruction = (
        "You are a helpful, professional, and knowledgeable AI assistant. You answer user questions "
        "using the provided context documents. Adhere strictly to the following rules:\n"
        "1. Base your answer ONLY on the provided Context below.\n"
        "2. If the Context does not contain the answer, say politely that you cannot find the answer "
        "in the uploaded documents, but do NOT make up any information.\n"
        "3. Maintain a professional, structured tone. Use markdown headings, bullet points, and bold text to organize your output.\n"
        "4. Always cite the exact source document name and page number (if available) when referring to facts.\n\n"
        f"--- CONTEXT ---\n{formatted_context}\n"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_instruction),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content=query)
    ])
    
    try:
        # Construct chain input
        chain_input = {
            "history": lc_history
        }
        
        # Format the prompt messages
        messages = prompt.format_messages(**chain_input)
        
        # Call LLM
        response = llm.invoke(messages)
        answer = response.content
        
        # Extract sources from documents for the frontend
        sources = []
        seen_sources = set()
        for doc in context_docs:
            src_name = doc.metadata.get("source", "Unknown")
            page_num = doc.metadata.get("page", None)
            source_key = f"{src_name}_page_{page_num}" if page_num else src_name
            
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "name": src_name,
                    "page": page_num,
                    "snippet": doc.page_content[:200] + "..."  # Short preview snippet
                })
                
        return answer, sources
        
    except Exception as e:
        logger.error(f"Error generating RAG answer: {str(e)}")
        raise RuntimeError(f"Failed to generate answer from Gemini: {str(e)}")

def generate_general_answer(
    query: str,
    chat_history: List[Dict[str, str]],
    google_api_key: str = None
) -> str:
    """
    Generates a general conversational response when no documents are uploaded.
    """
    llm = get_llm(google_api_key=google_api_key)
    lc_history = format_chat_history(chat_history)
    
    system_instruction = (
        "You are a helpful, conversational, and polite AI chatbot. You can answer general queries. "
        "If you do not know the answer, say so honestly. Do not make up information. Use clean Markdown formatting."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_instruction),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content=query)
    ])
    
    try:
        messages = prompt.format_messages(history=lc_history)
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error generating general answer: {str(e)}")
        raise RuntimeError(f"Failed to generate response: {str(e)}")
