from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
) 

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options.lang = ["rus", "eng"]
pipeline_options.do_table_structure = False
pipeline_options.generate_page_images = False

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )  
    }
)


def load_document(file_path: str):
    path = Path(file_path)
    result = converter.convert(str(path))
    return result.document.export_to_markdown()