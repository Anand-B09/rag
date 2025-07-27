import streamlit as st
from typing import Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
from datetime import datetime
import time

from .config import Settings
from .models import QueryResponse, DocumentMetadata, HealthResponse

logger = logging.getLogger(__name__)

class APIClient:
    """Handle all API communication with retry and error handling."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create a session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.settings.max_retries,
            backoff_factor=self.settings.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Make an API request with error handling and retries."""
        url = f"{self.settings.backend_host}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.settings.connection_timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            st.error(f"API request failed: {str(e)}")
            return None
            
    def get_health(self) -> Optional[HealthResponse]:
        """Check backend health status."""
        return self.request("GET", "health")
        
    def get_documents(self) -> Optional[List[DocumentMetadata]]:
        """Get list of all documents."""
        response = self.request("GET", "documents")
        return response.get("documents") if response else None
        
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        response = self.request("DELETE", f"documents/{doc_id}")
        return bool(response and response.get("status") == "success")
        
    def query_documents(self, query: str) -> Optional[QueryResponse]:
        """Query the documents."""
        return self.request("POST", "query", json={"query": query})
        
    def upload_document(self, file) -> Optional[DocumentMetadata]:
        """Upload a document."""
        return self.request(
            "POST",
            "ingest",
            files={"file": file}
        )
