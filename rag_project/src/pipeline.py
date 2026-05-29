from pathlib import Path 
from src.loader import load_document
from src.cleaner import clean_text
from src.chunker import chunk_text
def process_document(file_path: str): 
    raw_text = load_document(file_path) 
    cleaned_text = clean_text(raw_text) 
    chunks = chunk_text(cleaned_text) 
    documents = [] 
    for idx, chunk in enumerate(chunks): 
        documents.append( 
            { 
                "text": chunk, 
                "metadata": 
                { 
                    "source": Path(file_path).name, 
                    "page": None, 
                    "section": None, 
                    "chunk_id": idx, 
                }, 
            } 
        ) 
    return documents