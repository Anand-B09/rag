"""Integration tests for the RAG system."""
import pytest
from pathlib import Path
from unittest.mock import patch
from frontend.main import RAGApp
from frontend.ingest import process_file, process_batch

def test_document_ingestion_flow(mock_api_client, mock_chroma_client, test_pdf_path):
    """Test the complete document ingestion flow."""
    app = RAGApp()
    app.api_client = mock_api_client
    
    # First verify backend health
    app._check_backend_health()
    assert st.session_state.backend_healthy
    
    # Process the test PDF
    docs, success = process_file(test_pdf_path)
    assert success
    assert len(docs) > 0
    
    # Upload to mock backend
    class MockFile:
        name = Path(test_pdf_path).name
        
    app._upload_documents([MockFile()])
    mock_api_client.upload_document.assert_called_once()
    
    # Verify document appears in list
    app._refresh_documents()
    assert len(st.session_state.documents) > 0

def test_query_flow(mock_api_client):
    """Test the complete query flow."""
    app = RAGApp()
    app.api_client = mock_api_client
    st.session_state.backend_healthy = True
    
    # Submit a query
    test_query = "What is in the test document?"
    app._query_documents(test_query)
    
    # Verify chat history updated
    assert len(st.session_state.chat_history) == 1
    last_message = st.session_state.chat_history[-1]
    assert last_message["query"] == test_query
    assert "source_documents" in last_message
    
    # Verify API calls
    mock_api_client.query_documents.assert_called_with(test_query)
