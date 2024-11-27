"""Code to ingest blob into a vectorstore.

Code is responsible for taking binary data, parsing it and then indexing it
into a vector store.

This code should be agnostic to how the blob got generated; i.e., it does not
know about server/uploading etc.
"""
from typing import List

from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders import Blob
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def _update_document_metadata(document: Document, namespace: str, file_info: dict) -> None:
    """Update document metadata with namespace and file information."""
    document.metadata.update({
        "namespace": namespace,
        "file_name": file_info["name"],
        "file_type": file_info["type"],
        "file_size": file_info["size"],
        "chunk_index": file_info.get("chunk_index", 0),
        "total_chunks": file_info.get("total_chunks", 1),
        "is_file_header": file_info.get("is_file_header", True)
    })


def _sanitize_document_content(document: Document) -> Document:
    """Sanitize the document."""
    # Without this, PDF ingestion fails with
    # "A string literal cannot contain NUL (0x00) characters".
    document.page_content = document.page_content.replace("\x00", "x")


# PUBLIC API


def ingest_blob(
    blob: Blob,
    parser: BaseBlobParser,
    text_splitter: TextSplitter,
    vectorstore: VectorStore,
    namespace: str,
    *,
    batch_size: int = 100,
) -> List[str]:
    """Ingest a document into the vectorstore."""
    docs_to_index = []
    ids = []
       
    # Parse the document
    for document in parser.lazy_parse(blob):
        # Create file header
        header_doc = Document(
            page_content=f"File: {blob.path}",
            metadata={"namespace": namespace, "file_name": blob.path}
        )
        docs_to_index.append(header_doc)
        
        # Add the main document
        document.metadata.update({"namespace": namespace, "file_name": blob.path})
        _sanitize_document_content(document)
        docs_to_index.append(document)
            
        if len(docs_to_index) >= batch_size:
            ids.extend(vectorstore.add_documents(docs_to_index))
            docs_to_index = []

    if docs_to_index:
        ids.extend(vectorstore.add_documents(docs_to_index))

    return ids
