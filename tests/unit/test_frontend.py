"""Unit tests for frontend components."""
import pytest
from unittest.mock import patch
import streamlit as st
from datetime import datetime
from frontend.main import RAGApp

def test_check_backend_health_success(mock_api_client):
    """Test successful backend health check."""
    app = RAGApp()
    app.api_client = mock_api_client
    
    app._check_backend_health()
    assert st.session_state.backend_healthy == True

def test_check_backend_health_failure(mock_api_client):
    """Test failed backend health check."""
    app = RAGApp()
    app.api_client = mock_api_client
    mock_api_client.get_health.return_value = {"status": "unhealthy"}
    
    app._check_backend_health()
    assert st.session_state.backend_healthy == False

def test_check_backend_health_exception(mock_api_client):
    """Test backend health check with exception."""
    app = RAGApp()
    app.api_client = mock_api_client
    mock_api_client.get_health.side_effect = Exception("Connection error")
    
    app._check_backend_health()
    assert st.session_state.backend_healthy == False

def test_upload_documents_success(mock_api_client):
    """Test successful document upload."""
    app = RAGApp()
    app.api_client = mock_api_client
    st.session_state.backend_healthy = True
    
    class MockFile:
        name = "test.pdf"
    
    app._upload_documents([MockFile()])
    mock_api_client.upload_document.assert_called_once()

def test_query_documents_success(mock_api_client):
    """Test successful document query."""
    app = RAGApp()
    app.api_client = mock_api_client
    st.session_state.backend_healthy = True
    st.session_state.chat_history = []
    
    app._query_documents("test query")
    
    assert len(st.session_state.chat_history) == 1
    assert st.session_state.chat_history[0]["query"] == "test query"
