from typing import Dict, List, Optional, TypedDict
from datetime import datetime

class DocumentMetadata(TypedDict):
    id: str
    filename: str
    timestamp: str
    page_count: int
    status: str
    embedding_model: str

class ChatMessage(TypedDict):
    query: str
    response: str
    timestamp: str
    source_documents: List[DocumentMetadata]

class QueryResponse(TypedDict):
    response: str
    source_documents: List[DocumentMetadata]

class HealthResponse(TypedDict):
    status: str
    version: str
    components: Dict[str, str]
