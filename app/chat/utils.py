
from app.models.schema import (
    Document as DocumentSchema,
)

from llama_index.schema import (
  Document as LLamaIndexDocument
)
from pathlib import Path
from llama_index.readers.file.docs_reader import PDFReader

from app.chat.constants import (
    DOC_ID_KEY
)

from typing import Dict, List, Optional


def build_description_for_document(document: DocumentSchema) -> str:
    if document.metadata is not None:
        metadata = document.metadata
        return f"{metadata.document_description}"
    return "一份包含有用信息的文档。"

def build_title_for_document(document: DocumentSchema) -> str:
    if document.metadata is not None:
        return f"{document.metadata.document_description}"
    return "没有标题"


def fetch_and_read_documents(
        documents: List[DocumentSchema]) -> List[LLamaIndexDocument]:
    loaded_documents = []
    for doc in documents:
        reader = PDFReader()
        loaded = reader.load_data(
            Path(doc.url), extra_info={DOC_ID_KEY: str(doc.id)})
        loaded_documents.extend(loaded)
    return loaded_documents