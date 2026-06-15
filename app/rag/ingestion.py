"""
Document ingestion pipeline.

Flow:   
        PDF file - text extraction (pyMuPDF) - recursive chunking - embedding generation - ChromaDB upsert.
"""

import uuid
from pathlib import Path

import fitz # pyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.rag.vector_store import VectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)
 


def extract_text_from_pdf(file_path:str | Path) -> list[dict]:
    """
    Extract the text page by page from pdf  using pyMuPDF
    args: file_path : path to the pdf file.
    return : List of dicts with 'text' and "page" keys.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found {file_path}")


    pages:list[dict] = []
    
    with fitz.open(str(path)) as doc:
        for page_num,page in enumerate(doc,start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({"text":text,"page_num":page_num})    
                
    logger.info("PDF pages extracted",path = str(path),pages = len(pages))
    
    return pages


def chunk_pages(pages:list[dict],
                source_name:str,
                document_id:str,)-> tuple[list[str],list[dict],list[str]]:
    """
    recursively split page texts into overlapping chunks.
    Args: 
        pages: Output from extract_text_from_pdf .
        source_name:Human readable documente name (stored in metadata).  
        document_id : Database Document UUID (stored in metadata).
        return : 
                Tuple of (texts, metadatas,ids) ready for ChromaDB upsert.
    """
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = settings.CHUNK_SIZE,
        chunk_overlap = settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    texts : list[dict] = []
    metadatas : list[dict] = []
    ids : list[str] = []
    
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            texts.append(chunk)
            metadatas.append(
                {
                    "document_id" : document_id,
                    "source":source_name,
                    "page":page["page"],
                }
            )
            
            ids.append(chunk_id)
            
    logger.info("Chunks created",   
                document_id = document_id,
                chunks = len(texts))

    
    return texts, metadatas,ids



def ingest_pdf(
    file_path:str|Path,
    document_id : str,
    source_name : str | None = None,
) -> int:
    
    """
    Full ingestion pipeline for a single PDF.
    Args: 
        file_path : Path to PDF.
        document_id : Database UUID for this document.
        source_name : display name (defaults to file stem)
        
    return: 
            Number of chunks ingested.
    
    """
    
    path = Path(file_path)
    
    name = source_name or path.stem

    pages = extract_text_from_pdf(path)
    
    if not pages:
        raise ValueError("PDF contains no extractable texts.")
    
    texts, metadatas, ids  = chunk_pages(pages,name,document_id)
    store = VectorStore()
    
    counts = store.add_documents(texts,metadatas,ids)
    
    logger.info("Ingestion complete",document_id= document_id,chunks = counts)
    
    return counts
    