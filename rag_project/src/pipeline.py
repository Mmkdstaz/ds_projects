from pathlib import Path
from docling.chunking import HybridChunker

from src.cleaner import clean_text
from src.loader import load_document

NOISE_PATTERNS = [
    r"copyright",
    r"все права",
    r"запрещается.*воспроизведение",
    r"чтобы заказать копии",
    r"отправьте электронное сообщение",
    r"^ред\.\s*\d",
    r"редакция \d{1,2} \w+ \d{4}",
]

import re

def is_noise_chunk(text: str) -> bool:
    text_lower = text.lower()
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    if len(text.strip()) < 30:
        return True
    words = text.split()
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len > 15:  
            return True
    return False


def process_document(file_path: str):
    doc = load_document(file_path)

    chunker = HybridChunker(chunk_size=500, chunk_overlap=50)
    doc_chunks = list(chunker.chunk(doc))

    documents = []

    for idx, chunk in enumerate(doc_chunks):
        raw_text = getattr(chunk, "text", "")
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            continue

        if is_noise_chunk(cleaned_text):
            continue

        page_numbers = []
        meta = getattr(chunk, "meta", None)

        if meta and getattr(meta, "doc_items", None):
            for item in meta.doc_items:
                prov = getattr(item, "prov", None)
                if prov:
                    for p in prov:
                        page_no = getattr(p, "page_no", None)
                        if page_no is not None:
                            page_numbers.append(page_no)

        page = sorted(set(page_numbers))[0] if page_numbers else 1
        section = "Введение"

        if meta and getattr(meta, "headings", None):
            section = meta.headings[-1]

        documents.append({
            "text": cleaned_text,
            "metadata": {
                "source": Path(file_path).name,
                "page": page,
                "section": section,
                "chunk_id": idx,
            },
        })

    return documents