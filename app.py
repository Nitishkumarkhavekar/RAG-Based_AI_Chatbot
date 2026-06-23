import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Ensure the source directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.loader import load_multiple_documents
from src.splitter import split_documents
from src.embeddings import get_embeddings_model
from src.vector_store import create_vectorstore, save_vectorstore, load_vectorstore, similarity_search
from src.chatbot import generate_answer, generate_general_answer
from src.utils import validate_api_key, get_file_size_formatted, save_uploaded_file, clean_directory

# ----------------------------------------------------
# 1. Page Configuration & Custom CSS (Rich Aesthetics)
# ----------------------------------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Inject Custom Premium Styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Background Override */
    .stApp {
        background-color: #0B0C10 !important;
        color: #E2E8F0 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Top Navigation bar customization if present */
    header {
        background: rgba(11, 12, 16, 0.8) !important;
        backdrop-filter: blur(10px);
        border-bottom: 2px solid #1F222F !important;
    }

    /* Main Title Custom Styling with THICK BORDER */
    .hero-title-container {
        text-align: center !important;
        padding: 2.5rem 1.5rem;
        background: #12141D;
        border: 3px solid #262938 !important;
        border-radius: 16px;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Force centering on all sub-elements inside the container */
    .hero-title-container * {
        text-align: center !important;
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #6366F1 0%, #06B6D4 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-size: 3.5rem !important;
        line-height: 1.1;
        letter-spacing: -0.04em;
        margin: 0;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.15rem;
        font-weight: 400;
        margin-top: 0.75rem;
        max-width: 100% !important; /* Allow description to fit fully on a single line on desktop */
        margin-left: auto;
        margin-right: auto;
        text-align: center !important;
    }

    /* Sidebar Custom Styling with THICK BORDER */
    section[data-testid="stSidebar"] {
        background-color: #10121A !important;
        border-right: 3px solid #232736 !important;
    }
    
    /* Ensure all text inside sidebar is light and readable */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4, 
    section[data-testid="stSidebar"] h5, 
    section[data-testid="stSidebar"] h6,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] div {
        color: #F1F5F9 !important;
    }
    
    /* Apply borders to general widgets in the sidebar */
    section[data-testid="stSidebar"] .element-container {
        border-radius: 8px;
    }
    
    /* Custom divider/section headers in sidebar */
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    /* Chat Elements Custom Styling with THICK BORDERS */
    .stChatMessage {
        border-radius: 14px !important;
        padding: 1.25rem !important;
        margin-bottom: 1.25rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* User message card with thick Slate Blue border */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #141724 !important;
        border: 3px solid #4F46E5 !important;
    }
    .stChatMessage[data-testid="stChatMessageUser"]:hover {
        border-color: #6366F1 !important;
        box-shadow: 0 8px 30px rgba(79, 70, 229, 0.2) !important;
        transform: translateY(-1px);
    }
    
    /* Assistant message card with thick Deep Teal border */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #0E1117 !important;
        border: 3px solid #0D9488 !important;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"]:hover {
        border-color: #14B8A6 !important;
        box-shadow: 0 8px 30px rgba(13, 148, 216, 0.15) !important;
        transform: translateY(-1px);
    }

    /* Citation Block styling */
    .source-header {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #14B8A6;
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .sources-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 0.5rem;
    }
    
    .source-pill {
        background: rgba(20, 184, 166, 0.08);
        border: 2px solid #0D9488 !important;
        color: #2DD4BF;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        transition: all 0.2s ease;
    }
    
    .source-pill:hover {
        background: rgba(20, 184, 166, 0.15);
        border-color: #14B8A6 !important;
        color: #F8FAFC;
        transform: translateY(-1px);
    }

    /* Streamlit widgets border styling (Selectbox, file uploads) */
    .stSelectbox>div>div, .stFileUploader>div {
        border: 2px solid #2B2E3C !important;
        border-radius: 8px !important;
        background-color: #12141D !important;
        transition: border-color 0.2s ease;
    }
    
    .stSelectbox>div>div:focus-within {
        border-color: #6366F1 !important;
    }

    /* Style the text input box (API Key) to be white with black text */
    .stTextInput>div>div {
        border: 2px solid #CFD8DC !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important; /* White background */
        transition: border-color 0.2s ease;
    }
    
    .stTextInput>div>div:focus-within {
        border-color: #1A73E8 !important; /* Google Blue focus */
    }
    
    .stTextInput input {
        color: #000000 !important; /* Black text */
    }

    /* Button custom hover effects with borders */
    button {
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    button[kind="primary"] {
        background: #4F46E5 !important;
        border: 2px solid #6366F1 !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    button[kind="primary"]:hover {
        background: #5850EC !important;
        border-color: #818CF8 !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3) !important;
    }
    
    button[kind="secondary"] {
        background: #12141D !important;
        border: 2px solid #2B2E3C !important;
        color: #E2E8F0 !important;
    }
    
    button[kind="secondary"]:hover {
        border-color: #4F46E5 !important;
        color: #FFFFFF !important;
    }
    
    /* Styled lists of processed files with thick border */
    .file-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #141724;
        border: 2px solid #2E3245;
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Status Badges with thick border */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .status-success {
        background-color: rgba(16, 185, 129, 0.08);
        color: #34D399;
        border: 2.5px solid #10B981 !important;
    }
    .status-warning {
        background-color: rgba(245, 158, 11, 0.08);
        color: #FBBF24;
        border: 2.5px solid #F59E0B !important;
    }
    
    /* Chat Input Area border override */
    .stChatInputContainer, div[data-testid="stChatInput"] {
        border: 3px solid #CFD8DC !important; /* Professional light grey border */
        border-radius: 28px !important; /* Fully rounded capsule shape like Gemini */
        background-color: #F0F4F9 !important; /* Gemini light grey input background */
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    
    .stChatInputContainer:focus-within, div[data-testid="stChatInput"]:focus-within {
        border-color: #1A73E8 !important; /* Google Blue focus border */
        box-shadow: 0 4px 12px rgba(26, 115, 232, 0.15) !important;
    }

    /* Inner input text box override - BLACK text, Gemini font */
    .stChatInputContainer textarea, div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #000000 !important; /* BLACK text when typing */
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        border: none !important;
    }
    
    /* Style placeholder text for light background */
    .stChatInputContainer textarea::placeholder, div[data-testid="stChatInput"] textarea::placeholder {
        color: #5F6368 !important; /* Medium grey placeholder */
        opacity: 1 !important;
    }

    /* Bottom sticky container background color override (replaces white bar with dark background) */
    div[data-testid="stBottom"], [data-testid="stBottom"] > div {
        background-color: #0B0C10 !important;
        border-top: none !important;
    }

    /* Ensure all text inside chat messages (User & Assistant) is white/light grey */
    .stChatMessage, .stChatMessage p, .stChatMessage li, .stChatMessage span, .stChatMessage div {
        color: #E2E8F0 !important;
    }

    /* File Uploader and Uploaded File List text color override to be black/dark charcoal (since its background is white) */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] *,
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] div,
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] span,
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] p,
    section[data-testid="stSidebar"] div[data-testid="stUploadedFile"] *,
    section[data-testid="stSidebar"] div[data-testid="stUploadedFile"] div,
    section[data-testid="stSidebar"] div[data-testid="stUploadedFile"] span,
    section[data-testid="stSidebar"] div[data-testid="stUploadedFile"] p {
        color: #1E1F20 !important;
    }
    
    /* Ensure the upload button text remains white */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button *, 
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] [role="button"] *,
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] [role="button"] {
        color: #FFFFFF !important;
    }

    /* Force the widget label 'Upload Documents' and helper icons to remain white */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] label,
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] label * {
        color: #F1F5F9 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Path Setup
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# Ensure folders exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# ----------------------------------------------------
# 3. Session State Initialization
# ----------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Scan existing local files inside data dir on startup
existing_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
if not st.session_state.processed_files and existing_files:
    st.session_state.processed_files = existing_files

# ----------------------------------------------------
# 4. Sidebar Section
# ----------------------------------------------------
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
    st.markdown("### 🤖 RAG Engine Control", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 4.1 API Key Configuration
    st.markdown("#### 🔑 API Authentication", unsafe_allow_html=True)
    env_api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    # Text input for manual key override if needed
    user_api_key = st.text_input(
        "Google Gemini API Key",
        value=env_api_key if env_api_key else "",
        type="password",
        placeholder="Enter your Gemini API key...",
        help="You can get an API key from Google AI Studio."
    )
    
    # Active API key selection
    active_api_key = user_api_key if user_api_key else env_api_key
    
    # API key status indicator
    if active_api_key:
        # Validate the key lazily (or on changes) using cache/session
        is_valid = validate_api_key(active_api_key)
        if is_valid:
            # Set the environment variable for libraries
            os.environ["GOOGLE_API_KEY"] = active_api_key
            st.markdown(
                '<div class="status-badge status-success">● API Key Verified</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="status-badge status-warning">▲ Key Invalid / Quota Exceeded</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="status-badge status-warning">● No API Key Provided</div>',
            unsafe_allow_html=True
        )
        
    st.markdown("<hr style='border-color: rgba(255,255,255,0.06);'/>", unsafe_allow_html=True)
    
    # 4.2 Document Upload & Processing
    st.markdown("#### 📁 Knowledge Base", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or DOCX files to expand the chatbot's knowledge base."
    )
    
    # Trigger RAG processing
    process_clicked = st.button("Process Documents", type="primary", use_container_width=True)
    
    # Check if there's an existing index to load on startup
    if st.session_state.vectorstore is None and active_api_key:
        try:
            embeddings_model = get_embeddings_model(google_api_key=active_api_key)
            db = load_vectorstore(VECTOR_STORE_DIR, embeddings_model)
            if db:
                st.session_state.vectorstore = db
                st.success("Successfully loaded local vector index on startup.")
        except Exception as e:
            # Silent fail or minor warning on startup reload
            pass
            
    if process_clicked:
        if not active_api_key:
            st.error("Please provide a valid Gemini API Key to process documents.")
        elif not uploaded_files:
            st.warning("Please upload one or more files first.")
        else:
            with st.spinner("Initializing pipeline & loading files..."):
                try:
                    # 1. Clean the old local files and index if loading a new set
                    clean_directory(DATA_DIR)
                    clean_directory(VECTOR_STORE_DIR)
                    st.session_state.processed_files = []
                    
                    # 2. Save uploaded files to the local data directory
                    local_filepaths = []
                    for f in uploaded_files:
                        path = save_uploaded_file(f, DATA_DIR)
                        local_filepaths.append(path)
                        st.session_state.processed_files.append(f.name)
                    
                    # 3. Load text documents
                    st.text("Extracting document contents...")
                    documents = load_multiple_documents(uploaded_files)
                    
                    # 4. Split document chunks
                    st.text("Splitting text into context chunks...")
                    chunks = split_documents(documents)
                    
                    # 5. Generate embeddings and create vector store
                    st.text("Generating embeddings & building FAISS index...")
                    embeddings_model = get_embeddings_model(google_api_key=active_api_key)
                    vectorstore = create_vectorstore(chunks, embeddings_model)
                    
                    # 6. Save vector store locally
                    save_vectorstore(vectorstore, VECTOR_STORE_DIR)
                    
                    st.session_state.vectorstore = vectorstore
                    st.success("Processing completed! Vector database updated.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Pipeline error: {str(e)}")
                    
    # 4.3 Chat Mode Selection
    st.markdown("<hr style='border-color: rgba(255,255,255,0.06);'/>", unsafe_allow_html=True)
    st.markdown("#### 💬 Chat Mode", unsafe_allow_html=True)
    if st.session_state.processed_files:
        chat_mode = st.radio(
            "Select Chat Mode",
            ["Document Q&A (RAG)", "General Chat (No Docs)"],
            index=0,
            help="Switch between querying uploaded documents or general conversation."
        )
    else:
        chat_mode = "General Chat (No Docs)"
        st.info("Using General Chat mode (no documents uploaded).")

    # 4.4 Database Management and Reset
    st.markdown("<hr style='border-color: rgba(255,255,255,0.06);'/>", unsafe_allow_html=True)
    st.markdown("#### ⚙️ Database Controls", unsafe_allow_html=True)
    
    if st.session_state.processed_files:
        st.markdown("**Processed Documents:**")
        for filename in st.session_state.processed_files:
            st.markdown(f'<div class="file-item"><span>📄 {filename}</span></div>', unsafe_allow_html=True)
            
        clear_db = st.button("Reset Knowledge Base", use_container_width=True)
        if clear_db:
            clean_directory(DATA_DIR)
            clean_directory(VECTOR_STORE_DIR)
            st.session_state.vectorstore = None
            st.session_state.processed_files = []
            st.success("Knowledge base reset successfully.")
            st.rerun()
    else:
        st.info("No documents indexed. Using fallback general chat.")

    # 4.4 Utility Links / Help info
    st.markdown("<hr style='border-color: rgba(255,255,255,0.06);'/>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; color: #64748B; font-size: 0.8rem;'>"
        "Designed by Nitishkumar<br/>"
        
        "</div>",
        unsafe_allow_html=True
    )

# ----------------------------------------------------
# 5. Main Chat Area
# ----------------------------------------------------
# Hero Header Banner
st.markdown("""
<div class="hero-title-container">
    <h1 class="hero-title">RAG Chatbot</h1>
    <p class="hero-subtitle">Upload PDFs, TXT, or DOCX documents to query your knowledge base using high-precision semantic retrieval and Google Gemini.</p>
</div>
""", unsafe_allow_html=True)

# Warning if API key is missing
if not active_api_key:
    st.warning("👈 Please enter a Google Gemini API Key in the sidebar to start chatting.")

# Clear conversation history button
col1, col2 = st.columns([8, 2])
with col2:
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# 5.1 Display Chat Messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display sources if assistant returned them
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            st.markdown('<div class="source-header">Sources & Citations:</div>', unsafe_allow_html=True)
            st.markdown('<div class="sources-list">', unsafe_allow_html=True)
            for src in msg["sources"]:
                page_str = f" | Page {src['page']}" if src.get("page") else ""
                clean_snippet = src.get("snippet", "").replace('"', '&quot;').replace("'", '&apos;').replace('\n', ' ').replace('\r', ' ')
                st.markdown(
                    f'<span class="source-pill" title="{clean_snippet}">'
                    f'📖 {src["name"]}{page_str}'
                    f'</span>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

# 5.2 Chat Input & Generation
chat_placeholder = "Ask a question about your documents..." if chat_mode == "Document Q&A (RAG)" else "Ask Gemini a general question..."
if prompt := st.chat_input(chat_placeholder, disabled=not active_api_key):
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Append to session state
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # 2. Generate assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Setup loading spinner
        with st.spinner("Analyzing and retrieving context..."):
            try:
                # Route RAG vs General chat based on selected mode & vector store existence
                if chat_mode == "Document Q&A (RAG)" and st.session_state.vectorstore:
                    # RAG path
                    # Retrieve relevant chunks (k=4 by default)
                    retrieved_chunks = similarity_search(st.session_state.vectorstore, prompt, k=4)
                    
                    # Generate RAG answer
                    answer, sources = generate_answer(
                        query=prompt,
                        context_docs=retrieved_chunks,
                        chat_history=st.session_state.chat_history[:-1],  # Exclude current prompt
                        google_api_key=active_api_key
                    )
                    
                    # Render response
                    response_placeholder.markdown(answer)
                    
                    # Render sources
                    if sources:
                        st.markdown('<div class="source-header">Sources & Citations:</div>', unsafe_allow_html=True)
                        st.markdown('<div class="sources-list">', unsafe_allow_html=True)
                        for src in sources:
                            page_str = f" | Page {src['page']}" if src.get("page") else ""
                            clean_snippet = src.get("snippet", "").replace('"', '&quot;').replace("'", '&apos;').replace('\n', ' ').replace('\r', ' ')
                            st.markdown(
                                f'<span class="source-pill" title="{clean_snippet}">'
                                f'📖 {src["name"]}{page_str}'
                                f'</span>',
                                unsafe_allow_html=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    # Save assistant message to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    # Fallback General Chat path
                    answer = generate_general_answer(
                        query=prompt,
                        chat_history=st.session_state.chat_history[:-1],
                        google_api_key=active_api_key
                    )
                    response_placeholder.markdown(answer)
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": []
                    })
                    
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                response_placeholder.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })
