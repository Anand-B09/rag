from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel, Field, BaseSettings
import chromadb
import tempfile
import os
import fitz  # PyMuPDF
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import uvicorn
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
# Configuration
class Settings(BaseSettings):
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    embed_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "llama3.1:latest"
    collection_name: str = "pdf_docs"
    
    class Config:
        env_prefix = "RAG_"

@lru_cache()
def get_settings():
    return Settings()

class DocumentMetadata(BaseModel):
    id: str
    filename: str
    timestamp: str
    page_count: int
    status: str = "active"
    source: str
    embedding_model: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)

class QueryResponse(BaseModel):
    response: str
    source_documents: List[DocumentMetadata]

class HealthStatus(BaseModel):
    status: str
    version: str = "1.0.0"
    components: Dict[str, str]

class RAGService:
    def __init__(self, settings: Settings = Depends(get_settings)):
        """Initialize RAG service with configuration."""
        self.settings = settings
        self.documents: Dict[str, DocumentMetadata] = {}
        self.embed_model = None
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize ChromaDB, vector store, and LLM components."""
        try:
            # Initialize embedding model
            self.embed_model = HuggingFaceEmbedding(
                model_name=self.settings.embed_model
            )
            Settings.embed_model = self.embed_model  # Set global embed model
            
            # Initialize ChromaDB
            self.client = chromadb.HttpClient(
                host=self.settings.chroma_host,
                port=self.settings.chroma_port
            )
            
            # Create or get collection with metadata
            self.collection = self.client.get_or_create_collection(
                name=self.settings.collection_name,
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "embedding_model": self.settings.embed_model,
                    "llm_model": self.settings.llm_model
                }
            )
            
            # Initialize vector store and index
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )
            
            # Initialize LLM
            self.llm = Ollama(
                model=self.settings.llm_model,
                request_timeout=300.0
            )
            
            # Create query engine
            self.query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=5
            )
            
            # Load existing documents
            self.load_existing_documents()
            logger.info("RAG Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Service: {str(e)}")
            raise
    
    def load_existing_documents(self):
        """Load existing documents metadata from ChromaDB."""
        try:
            metadata_list = self.collection.get()
            for doc_id, metadata in zip(metadata_list.get("ids", []), metadata_list.get("metadatas", [])):
                if metadata:
                    self.documents[doc_id] = DocumentMetadata(
                        id=doc_id,
                        filename=metadata.get("filename", "unknown"),
                        timestamp=metadata.get("timestamp", datetime.now().isoformat()),
                        page_count=metadata.get("page_count")
                    )
            logger.info(f"Loaded {len(self.documents)} existing documents")
        except Exception as e:
            logger.error(f"Error loading existing documents: {str(e)}")
            raise

    def extract_pdf_text(self, file_path: str) -> tuple[str, int]:
        """
        Extract text content from a PDF file.
        Returns tuple of (text, page_count).
        """
        try:
            doc = fitz.open(file_path)
            text = ""
            page_count = doc.page_count
            
            if page_count == 0:
                raise ValueError("PDF file contains no pages")
            
            for page in doc:
                page_text = page.get_text().strip()
                if page_text:  # Only add non-empty pages
                    text += page_text + "\n\n"
            
            doc.close()
            
            if not text.strip():
                raise ValueError("PDF file contains no extractable text")
                
            return text.strip(), page_count
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise

    def add_document_to_index(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add a document to the vector store and update tracking."""
        try:
            # Generate unique document ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            
            # Create document with metadata
            document = Document(
                text=text,
                metadata={
                    **metadata,
                    "id": doc_id,
                    "embedding_model": self.settings.embed_model,
                    "source": "pdf_upload",
                    "indexed_at": datetime.now().isoformat()
                }
            )
            
            # Insert into vector store
            self.index = self.index.insert(document)
            
            # Update tracking with full metadata
            self.documents[doc_id] = DocumentMetadata(
                id=doc_id,
                filename=metadata["filename"],
                timestamp=metadata["timestamp"],
                page_count=metadata["page_count"],
                source=metadata.get("source", "pdf_upload"),
                embedding_model=self.settings.embed_model,
                status="active"
            )
            
            logger.info(f"Successfully added document {doc_id} to index")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document to index: {str(e)}")
            raise

    def delete_document_from_store(self, doc_id: str) -> bool:
        """Remove a document from the vector store and tracking."""
        try:
            if doc_id not in self.documents:
                return False
                
            # Mark document as deleted in ChromaDB
            self.collection.update(
                ids=[doc_id],
                metadatas=[{"status": "deleted", "deleted_at": datetime.now().isoformat()}]
            )
            
            # Remove from vector store
            self.collection.delete(ids=[doc_id])
            
            # Remove from tracking
            doc_info = self.documents.pop(doc_id)
            logger.info(f"Successfully deleted document {doc_id} ({doc_info.filename})")
            
            # Force refresh index
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise

    def query_documents(self, query: str, top_k: int = 5) -> QueryResponse:
        """Query the document store and return response with sources."""
        try:
            response = self.query_engine.query(query)
            source_docs = [
                self.documents[node.node.metadata["id"]]
                for node in response.source_nodes
                if node.node.metadata["id"] in self.documents
            ]
            return QueryResponse(
                response=str(response),
                source_documents=source_docs[:top_k]
            )
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            raise

    def get_health_status(self) -> HealthStatus:
        """Get the health status of all components."""
        status = {
            "chromadb": "healthy" if self.client.heartbeat() else "unhealthy",
            "llm": "healthy",  # Add actual health check if available
            "document_count": str(len(self.documents))
        }
        return HealthStatus(
            status="healthy" if all(s == "healthy" for s in status.values()) else "degraded",
            components=status
        )

# Initialize RAG Service
rag_service = RAGService()

# Initialize FastAPI app with CORS
app = FastAPI(title="RAG API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check() -> HealthStatus:
    """Check if the service is healthy."""
    return rag_service.get_health_status()

@app.get("/documents")
async def list_documents() -> Dict[str, List[DocumentMetadata]]:
    """List all processed documents."""
    return {"documents": list(rag_service.documents.values())}

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

@app.post("/query")
async def query_api(query_request: QueryRequest) -> QueryResponse:
    """Query the document store."""
    try:
        return rag_service.query_documents(
            query_request.query,
            query_request.top_k
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_api(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings)
) -> DocumentMetadata:
    """Ingest a PDF document."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    tmp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Extract text and process document
        text, page_count = rag_service.extract_pdf_text(tmp_path)
        
        # Prepare metadata
        current_time = datetime.now().isoformat()
        metadata = {
            "filename": file.filename,
            "timestamp": current_time,
            "page_count": page_count,
            "source": "pdf_upload",
            "file_size": len(content),
            "mime_type": file.content_type,
            "embedding_model": settings.embed_model
        }
        
        # Add to index
        doc_id = rag_service.add_document_to_index(
            text=text,
            metadata=metadata
        )
        
        logger.info(f"Successfully ingested {file.filename} ({page_count} pages)")
        return rag_service.documents[doc_id]
    
    except ValueError as e:
        logger.error(f"Invalid PDF content: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
