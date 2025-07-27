from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dataclasses import dataclass, asdict, field
from typing import TypedDict, Any, List, Dict, Optional
import os
from pydantic import BaseModel

import chromadb
import tempfile
import fitz  # PyMuPDF

from datetime import datetime
import logging
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Pydantic models
class QueryRequest(BaseModel):    
    query: str
    top_k: int = 5  # Validated in the API endpoint
    enable_streaming: bool = False


class QueryResponse(BaseModel):
    response: str
    sources: List[Dict]
    processing_time: float

class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    chroma_connected: bool
    documents_count: int

class DocumentInfo(BaseModel):
    filename: str
    page_count: int
    upload_time: str
    file_size: int


class RAGService:
    def __init__(self):
        self.documents_info = {}
        self.initialize_components()
        
    def initialize_components(self):
        """ Initialize ChromaDB, Ollama, and other components """
        try:
            # Initialize ChromaDB client
            chroma_host = os.getenv("CHROMA_HOST", "chromadb")
            chroma_port = os.getenv("CHROMA_PORT", "8000")
            self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            try:
                self.client.heartbeat()
                logger.info(f"Connected to ChromaDB successfully at {chroma_host}:{chroma_port}")
            except Exception as e:
                logger.warning(f"ChromaDB connection test failed: {e}")

            self.collection = self.client.get_or_create_collection(
                name="pdf_docs",
                metadata={
                    "description": "Collection for PDF documents",
                    "created_at": datetime.now().isoformat(),
                    "source": "rag_service",
                    "hnsw:space": "cosine"  # Use cosine similarity
                }
            )

            embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
            self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

            # Initialize LLM
            ollama_host = os.getenv("OLLAMA_HOST", "ollama")
            ollama_port = os.getenv("OLLAMA_PORT", "11434")
            llm_model = os.getenv("LLM_MODEL", "gemma3:1b")
            base_url = f"http://{ollama_host}:{ollama_port}"
            
            # Test Ollama connection first
            import requests
            try:
                health_response = requests.get(f"{base_url}")
                if health_response.status_code == 200:
                    logger.info(f"Ollama server is healthy at {base_url}")
                    
                    # Check if model is pulled
                    model_response = requests.get(f"{base_url}/api/tags")
                    if model_response.status_code == 200:
                        available_models = [m.get("name") for m in model_response.json().get("models", [])]
                        if llm_model not in available_models:
                            logger.warning(f"Model {llm_model} not found, it will be pulled on first use")
                    
                    self.llm = Ollama(
                        model=llm_model,
                        base_url=base_url,
                        request_timeout=300.0
                    )
                    logger.info(f"Initialized LLM: {llm_model} at {base_url}")
                else:
                    raise ConnectionError(f"Ollama server health check failed with status {health_response.status_code}")
            except requests.RequestException as e:
                logger.error(f"Failed to connect to Ollama server at {base_url}: {str(e)}")
                # Initialize anyway to allow startup, but log the error
                self.llm = Ollama(
                    model=llm_model,
                    base_url=base_url,
                    request_timeout=300.0
                )

            # Initialize Vector Store
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)

            # Initialize Storage Context
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            try:
                # Check if the collection has documents first
                collection_count = self.collection.count()
                if collection_count > 0:
                    self.index = VectorStoreIndex.from_vector_store(
                        self.vector_store, 
                        embed_model=self.embed_model
                    )
                    logger.info(f"Loaded existing index with {collection_count} documents")
                else:
                    self.index = None
                    logger.info("No existing documents found in ChromaDB, starting with an empty index")
            except Exception as e:
                logger.warning(f"Error loading existing index: {e}")
                self.index = None
                logger.info("Starting with an empty index")
            
            self.load_existing_documents()
            logger.info("RAG Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Service: {str(e)}")

   
    def load_existing_documents(self):
        try:
            """ Load existing documents from ChromaDB into the service """
            result = self.collection.get(include=["metadatas"])

            if result and result["metadatas"]:
                filename_stats = {}
                for metadata in result["metadatas"]:
                    if metadata and 'source' in metadata:
                        filename = metadata['source']
                        if filename not in filename_stats:
                            filename_stats[filename] = {
                                "page_count": 0,
                                "upload_time": metadata.get("upload_time", 'Unknown'),
                                "file_size": 'Unknown'
                            }
                        filename_stats[filename]["page_count"] += 1
                for filename, stats in filename_stats.items():
                    self.documents_info[filename] = {
                        "filename": filename,
                        "page_count": stats["page_count"],
                        "upload_time": stats["upload_time"],
                        "file_size": stats.get("file_size", 0) if isinstance(stats.get("file_size"), int) else 0
                    }
                logger.info(f"Loaded {len(filename_stats)} existing documents from ChromaDB")
            else:
                logger.info("No existing documents found in ChromaDB")
        except Exception as e:
            logger.error(f"Error loading existing documents: {str(e)}")


    def extract_pdf_text(self, pdf_path: str, filename: str) -> List[Document]:
        """
        Extract text from each page of a PDF and return a list of Document objects.
        Each Document contains the text and metadata for a single page.
        """
        documents = []
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF File not found: {pdf_path}")
            
            doc = fitz.open(pdf_path)            
            page_count = doc.page_count

            if page_count == 0:
                logger.warning(f"PDF {filename} contains no pages.")
                return documents

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text").strip()
                if page_text:
                    metadata = {
                        "filename": filename,
                        "page": page_num + 1,                        
                        "upload_time": datetime.now().isoformat(),
                        "source": filename
                    }
                    documents.append(Document(text=page_text, metadata=metadata))
            doc.close()

            if not documents:
                logger.warning(f"No extractable text found in {filename}.")

            return documents
        except Exception as e:
            logger.error(f"Error extracting PDF text from {filename}: {str(e)}")
            return documents

    def add_document_to_index(self, documents: List[Document]):
        """
        Add a list of Document objects to the vector store and update the index.
        """
        try:
            if not documents:
                logger.warning("No documents provided to add to index.")
                return

            # If index does not exist, create it
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    embed_model=self.embed_model,
                    storage_context=self.storage_context
                )
            else:
                for doc in documents:
                    self.index.insert(doc)
                logger.info(f"Added {len(documents)} documents to existing index")
            self.index.storage_context.persist()

        except Exception as e:
            logger.error(f"Error adding documents to index: {str(e)}")

    def delete_document_from_store(self, filename: str) -> bool:
        """
        Delete all pages of a document (by filename) from the vector store and update metadata.
        Returns True if any document was deleted, False otherwise.
        """
        try:
            # Get all document IDs for the given filename
            result = self.collection.get(include=["ids", "metadatas"])
            ids_to_delete = []
            if result and result["ids"] and result["metadatas"]:
                for doc_id, metadata in zip(result["ids"], result["metadatas"]):
                    if metadata and metadata.get("filename") == filename:
                        ids_to_delete.append(doc_id)

            if not ids_to_delete:
                logger.warning(f"No documents found for filename: {filename}")
                return False

            # Delete from ChromaDB collection
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} pages for document '{filename}' from ChromaDB.")

            # Remove from local metadata
            if filename in self.documents_info:
                del self.documents_info[filename]

            # Rebuild index if needed
            if self.index is not None:
                # Rebuild index from remaining documents
                remaining_result = self.collection.get(include=["metadatas"])
                remaining_docs = []
                if remaining_result and remaining_result["metadatas"]:
                    for metadata in remaining_result["metadatas"]:
                        if metadata and "text" in metadata:
                            remaining_docs.append(Document(text=metadata["text"], metadata=metadata))
                if remaining_docs:
                    self.index = VectorStoreIndex.from_documents(
                        remaining_docs,
                        embed_model=self.embed_model,
                        storage_context=self.storage_context
                    )
                else:
                    self.index = None

            return True
        except Exception as e:
            logger.error(f"Error deleting document '{filename}': {str(e)}")
            return False

    def query_documents(self, query: str, top_k: int = 5, enable_streaming: bool = False) -> Dict:
        """
        Query the vector index and return the top matching documents and LLM response.
        """
        import time
        start_time = time.time()
        if not self.index:
            logger.warning("No index available for querying.")
            return {
                "response": "",
                "sources": [],
                "processing_time": 0.0
            }
        try:
            logger.info(f"Querying documents with query: {query}, top_k: {top_k}, streaming: {enable_streaming}")
            # Query the index
            logger.info(f"LLM: {self.llm}")
            query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=top_k,
                streaming=enable_streaming
            )
            response = query_engine.query(query)
            logger.info(f"Query response: {response}")
            source_docs = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    source_info = {
                        "source": node.metadata.get("source", "Unknown"),
                        "page": node.metadata.get("page", "Unknown"),
                        "text": node.node.text[:300] if hasattr(node.node, 'text') else "",
                        "score": float(getattr(node, 'score', 0.0))
                    }
                    source_docs.append(source_info)
            processing_time = time.time() - start_time
            logger.info(f"Query processed in {processing_time:.2f} seconds")
            return {
                "response": str(response.response) if hasattr(response, 'response') else str(response),
                "sources": source_docs,
                "processing_time": processing_time
            }
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            return {
                "response": "",
                "sources": [],
                "processing_time": 0.0
            }

    def get_health_status(self) -> Dict:
        """Get service health status"""
        try:
            # Check ChromaDB connection
            max_retries = 3
            chroma_connected = False
            for attempt in range(max_retries):
                try:
                    self.client.heartbeat()
                    chroma_connected = True
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"ChromaDB connection test failed after {max_retries} attempts: {str(e)}")
                    else:
                        logger.warning(f"ChromaDB connection test failed: {str(e)}, retrying...")
                        import time
                        time.sleep(1)

            # Check Ollama connection
            ollama_connected = False
            try:
                import requests
                base_url = self.llm.base_url.rstrip('/')
                health_response = requests.get(f"{base_url}", timeout=5)
                ollama_connected = health_response.status_code == 200
            except Exception as e:
                logger.error(f"Ollama health check failed: {str(e)}")

            documents_count = len(self.documents_info) if self.documents_info else 0
            
            # Overall status is healthy only if both critical components are connected
            status = "healthy" if (chroma_connected and ollama_connected) else "degraded"
            
            return {
                "status": status,
                "chroma_connected": chroma_connected,
                "ollama_connected": ollama_connected,
                "documents_count": documents_count
            }
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {
                "status": "unhealthy",
                "chroma_connected": False,
                "ollama_connected": False,
                "documents_count": 0
            }

# Initialize RAG Service
rag_service = RAGService()

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """Ingest multiple PDF documents."""
    logger.info("Ingesting documents...")
    processed_count = 0
    start_time = datetime.now()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        all_documents = []

        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="File name is required")
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
            content = await file.read()
            if len(content) > 50 * 1024 * 1024: # 50MB limit
                raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
    
            tmp_path = None
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                # Extract text and process document
                documents = rag_service.extract_pdf_text(
                    pdf_path=tmp_path,
                    filename=file.filename
                )

                if not documents:
                    logger.warning(f"No text extracted from {file.filename}, skipping.")
                    continue

                all_documents.extend(documents)
                current_time = datetime.now().isoformat()
                rag_service.documents_info[file.filename] = {
                    "filename": file.filename,
                    "timestamp": current_time,
                    "page_count": len(documents),
                    "source": "pdf_upload",
                    "file_size": len(content),
                    "mime_type": file.content_type,
                }
                processed_count += 1
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # Add all documents to index
        if all_documents:
            rag_service.add_document_to_index(all_documents)

        processing_time = (datetime.now() - start_time).total_seconds()        
        logger.info(f"Processed {processed_count} documents in {processing_time:.2f} seconds")

        return IngestResponse(
            message=f"Successfully processed {processed_count} documents",
            documents_processed=processed_count,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is healthy."""
    return rag_service.get_health_status()

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all processed documents."""
    return list(rag_service.documents_info.values())

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the system."""
    try:
        if rag_service.delete_document_from_store(doc_id):
            return JSONResponse({"status": "success"})
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(query_request: QueryRequest):
    """Query the document store."""
    if not query_request.query or not query_request.query.strip():
        raise HTTPException(status_code=400, detail="Query field is required and cannot be empty")
    if len(query_request.query) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters long")
    if query_request.top_k < 1 or query_request.top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
    try:            
        result = rag_service.query_documents(
            query=query_request.query.strip(),
            top_k=query_request.top_k,
            enable_streaming=query_request.enable_streaming
        )
        return QueryResponse(**result)
    except KeyError:
        raise HTTPException(status_code=400, detail="query field is required")
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
