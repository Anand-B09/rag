"""
PDF Document Ingestion Script for RAG System.

This script handles the ingestion of PDF documents into a ChromaDB vector store for RAG applications.
It includes:
- PDF text extraction with validation
- Vector store management
- Batch processing with progress tracking
- Resource cleanup and error handling
"""

import argparse
import os
import fitz
import chromadb
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Optional, List, Dict, Generator, Any, Tuple
import logging
from datetime import datetime
from tqdm import tqdm

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ingest.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 1.0
BATCH_SIZE = 100
DEFAULT_CHUNK_SIZE = 1000

# Initialize embedding model
try:
    EMBED = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    logger.info("Successfully initialized embedding model")
except Exception as e:
    logger.error(f"Failed to initialize embedding model: {str(e)}")
    raise

def get_chroma_client(
    host: str = "chroma",
    port: int = 8000,
    max_retries: int = MAX_RETRIES
) -> chromadb.HttpClient:
    """
    Initialize and return a ChromaDB client with retry logic.
    
    Args:
        host: ChromaDB host address
        port: ChromaDB port number
        max_retries: Maximum number of connection attempts
    
    Returns:
        ChromaDB HTTP client instance
    
    Raises:
        ConnectionError: If unable to connect after max retries
    """
    for attempt in range(max_retries):
        try:
            client = chromadb.HttpClient(host=host, port=port)
            # Test connection
            client.heartbeat()
            logger.info(f"Successfully connected to ChromaDB at {host}:{port}")
            return client
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to connect to ChromaDB after {max_retries} attempts")
                raise ConnectionError(f"Could not connect to ChromaDB: {str(e)}")
            
            wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
            logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s...")
            time.sleep(wait_time)

def get_or_create_collection(
    client: chromadb.HttpClient,
    collection_name: str,
    metadata: Optional[dict] = None
) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection with metadata.
    
    Args:
        client: ChromaDB client instance
        collection_name: Name of the collection
        metadata: Optional metadata for the collection
    
    Returns:
        ChromaDB collection instance
    """
    try:
        default_metadata = {
            "description": "PDF document collection for RAG",
            "created_at": "",
            "updated_at": "",
            "document_count": 0
        }
        
        # Merge with provided metadata
        collection_metadata = {**default_metadata, **(metadata or {})}
        
        # Get existing or create new collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata=collection_metadata
        )
        
        logger.info(f"Successfully accessed collection: {collection_name}")
        return collection
    except Exception as e:
        logger.error(f"Error accessing collection {collection_name}: {str(e)}")
        raise

# Initialize ChromaDB client and collection
CHROMA = get_chroma_client()
COLL = get_or_create_collection(CHROMA, "pdf_docs", {
    "description": "PDF documents for RAG system",
    "created_at": "",
    "embedding_model": "BAAI/bge-small-en-v1.5"
})

def validate_pdf(path: str) -> Tuple[bool, str]:
    """
    Validate a PDF file before processing.
    
    Args:
        path: Path to the PDF file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not os.path.exists(path):
            return False, "File does not exist"
            
        if not os.path.isfile(path):
            return False, "Path is not a file"
            
        if not path.lower().endswith('.pdf'):
            return False, "File is not a PDF"
            
        # Check if file is readable and actually a PDF
        doc = fitz.open(path)
        if doc.page_count == 0:
            doc.close()
            return False, "PDF contains no pages"
            
        doc.close()
        return True, ""
        
    except Exception as e:
        return False, f"Invalid PDF file: {str(e)}"

def pdf_to_docs(path: str) -> Generator[Document, None, None]:
    """
    Convert PDF pages to Document objects.
    
    Args:
        path: Path to the PDF file
    
    Yields:
        Document objects for each page
    
    Raises:
        ValueError: If PDF processing fails
    """
    try:
        pdf = fitz.open(path)
        file_size = os.path.getsize(path)
        
        try:
            for page in pdf:
                text = page.get_text("text").strip()
                if text:  # Only yield non-empty pages
                    yield Document(
                        text=text,
                        metadata={
                            "source": os.path.basename(path),
                            "page": page.number + 1,
                            "total_pages": pdf.page_count,
                            "file_size": file_size,
                            "processed_at": datetime.now().isoformat()
                        }
                    )
        finally:
            pdf.close()
            
    except Exception as e:
        raise ValueError(f"Error processing PDF {path}: {str(e)}")

def process_batch(
    nodes: List[Document],
    vs: ChromaVectorStore,
    batch_size: int = BATCH_SIZE
) -> None:
    """
    Process a batch of documents.
    
    Args:
        nodes: List of documents to process
        vs: Vector store instance
        batch_size: Size of batches for processing
    """
    total_batches = (len(nodes) + batch_size - 1) // batch_size
    
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]
        batch_idx = VectorStoreIndex.from_documents(
            batch,
            embed_model=EMBED,
            storage_context=StorageContext.from_defaults(vector_store=vs)
        )
        batch_idx.storage_context.persist()
        logger.info(f"Processed batch {(i // batch_size) + 1}/{total_batches}")

def process_file(file_path: str) -> Tuple[List[Document], bool]:
    """
    Process a single PDF file.
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        Tuple of (list of documents, success status)
    """
    try:
        is_valid, error = validate_pdf(file_path)
        if not is_valid:
            logger.error(f"Invalid PDF {file_path}: {error}")
            return [], False
            
        docs = list(pdf_to_docs(file_path))
        if not docs:
            logger.warning(f"No extractable text found in {file_path}")
            return [], False
            
        return docs, True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return [], False

def main(folder: str, batch_size: int = BATCH_SIZE) -> None:
    """
    Main ingestion function.
    
    Args:
        folder: Path to folder containing PDFs
        batch_size: Number of documents to process in each batch
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"Invalid folder path: {folder}")
        
    pdf_files = list(folder_path.glob("**/*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {folder}")
        return
        
    nodes = []
    processed_files = 0
    failed_files = 0
    
    # Process files with progress bar
    with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
        with ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(process_file, str(file_path)): file_path
                for file_path in pdf_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    docs, success = future.result()
                    if success:
                        nodes.extend(docs)
                        processed_files += 1
                    else:
                        failed_files += 1
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {str(e)}")
                    failed_files += 1
                finally:
                    pbar.update(1)
    
    if not nodes:
        logger.warning("No documents were successfully processed")
        return
    
    # Update collection metadata
    current_time = datetime.now().isoformat()
    try:
        COLL.modify(metadata={
            "updated_at": current_time,
            "document_count": COLL.count() + len(nodes),
            "last_ingestion": current_time,
            "total_files_processed": processed_files,
            "failed_files": failed_files
        })
    except Exception as e:
        logger.error(f"Failed to update collection metadata: {str(e)}")
    
    # Process documents in batches
    try:
        vs = ChromaVectorStore(chroma_collection=COLL)
        process_batch(nodes, vs, batch_size)
        
        # Log summary
        logger.info(
            f"Ingestion complete:\n"
            f"- Processed files: {processed_files}\n"
            f"- Failed files: {failed_files}\n"
            f"- Total pages: {len(nodes)}\n"
            f"- Average pages per file: {len(nodes)/processed_files if processed_files else 0:.1f}"
        )
        
    except Exception as e:
        logger.error(f"Failed to ingest documents: {str(e)}")
        raise

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="pdfs")
    main(ap.parse_args().path)
