from pathlib import Path
from typing import List

from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

from ..models.schema import Document as DocumentSchema
from .constants import DOC_ID_KEY


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
    documents: List[DocumentSchema],
) -> List[Document]:
    loaded_documents = []
    for doc in documents:
        loaded_documents.extend(PyPDFLoader(doc.url).load())
    return loaded_documents
