import os
import sys
from dotenv import load_dotenv

# Add src folder to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loader import load_txt
from src.splitter import split_documents
from src.embeddings import get_embeddings_model
from src.vector_store import create_vectorstore, similarity_search

def run_test():
    print("=== Antigravity RAG Verification Test ===")
    
    # Load env variables
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[WARNING] GOOGLE_API_KEY environment variable not set. Real API tests will be skipped.")
        return
        
    print("[1/4] Creating test content...")
    test_content = (
        "Antigravity coding assistant is created by the Google DeepMind team. "
        "It excels at complex software engineering tasks, web application generation, "
        "and codebase editing. It has capabilities to define and run autonomous subagents."
    )
    
    # Create a mock file
    test_file_path = "test_doc.txt"
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)
        
    print(f"[2/4] Loading document '{test_file_path}'...")
    documents = load_txt(test_file_path, "test_doc.txt")
    print(f"Loaded {len(documents)} document(s).")
    
    print("[3/4] Chunking document...")
    chunks = split_documents(documents, chunk_size=200, chunk_overlap=20)
    print(f"Generated {len(chunks)} chunks.")
    
    print("[4/4] Generating embeddings and running similarity search...")
    embeddings_model = get_embeddings_model(google_api_key=api_key)
    vectorstore = create_vectorstore(chunks, embeddings_model)
    
    query = "Who created Antigravity?"
    results = similarity_search(vectorstore, query, k=1)
    
    print(f"Query: '{query}'")
    if results:
        print(f"Result found in source: {results[0].metadata['source']}")
        print(f"Content: {results[0].page_content}")
        print("\n=== SUCCESS: Verification complete! ===")
    else:
        print("\n=== FAILURE: No documents retrieved. ===")
        
    # Cleanup temp file
    if os.path.exists(test_file_path):
        os.remove(test_file_path)

if __name__ == "__main__":
    run_test()
