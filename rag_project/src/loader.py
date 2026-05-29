from pathlib import Path
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

def load_document(file_path: str):
    path = Path(file_path)

    result = converter.convert(str(path))

    return result.document.export_to_markdown()