import os
import sys
from dotenv import load_dotenv
if sys.platform == "win32":
    os.environ["TESSDATA_PREFIX"] = r"C:\Users\monitoring\AppData\Local\Programs\Tesseract-OCR\tessdata"

os.environ["OMP_THREAD_LIMIT"] = "1"
load_dotenv()

from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = TesseractCliOcrOptions(
    lang=["rus", "eng"],
    tesseract_cmd=r"C:\Users\monitoring\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
)
pipeline_options.do_table_structure = False
pipeline_options.generate_page_images = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options,
                                         num_threads = 2)
    }
)

def load_document(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)
    result = converter.convert(str(path))
    if not result or not result.document:
        raise ValueError("Empty document from Docling")
    return result.document