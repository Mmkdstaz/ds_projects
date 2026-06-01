import re
import logging
from pathlib import Path

from src.cleaner import clean_text
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.loader import load_document

logger = logging.getLogger(__name__)

NOISE_PATTERNS = [
    r"copyright",
    r"все права",
    r"запрещается.*воспроизведение",
    r"чтобы заказать копии",
    r"отправьте электронное сообщение",
    r"^ред\.\s*\d",
    r"редакция \d{1,2} \w+ \d{4}",
]

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


def is_noise_chunk(text: str) -> bool:
    text_lower = text.lower()
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    if len(text.strip()) < 50:
        return True
    words = text.split()
    if words and sum(len(w) for w in words) / len(words) > 15:
        return True
    return False


def process_document(file_path: str) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый формат: {path.suffix}")

    pages = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = []
    chunk_counter = 0

    for page_data in pages:
        chunks = splitter.split_text(page_data["text"])
        for chunk in chunks:
            cleaned = clean_text(chunk)
            if not cleaned or is_noise_chunk(cleaned):
                continue
            documents.append({
                "text": cleaned,
                "metadata": {
                    "source": path.name,
                    "page": page_data["page"],
                    "chunk_id": chunk_counter,
                },
            })
            chunk_counter += 1

    logger.info("%s — %d чанков", path.name, len(documents))
    return documents


def process_folder(folder: str) -> list[dict]:
    folder_path = Path(folder)
    all_docs = []

    files = [
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    logger.info("Файлов: %d", len(files))

    for file in files:
        try:
            docs = process_document(str(file))
            all_docs.extend(docs)
            logger.info("✓ %s — %d чанков", file.name, len(docs))
        except Exception as e:
            logger.error("✗ %s — %s", file.name, e)

    logger.info("Итого: %d чанков", len(all_docs))
    return all_docs