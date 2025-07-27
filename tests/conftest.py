"""Test configuration and fixtures."""
import os
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass

# Add the project root directory to Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class TestSettings:
    """Test settings with default values."""
    backend_host: str = "http://backend:8001"
    max_retries: int = 3
    retry_delay: float = 1.0
    chat_history_limit: int = 50
    connection_timeout: float = 10.0

@pytest.fixture
def test_settings():
    """Provide test settings."""
    return TestSettings()

@pytest.fixture
def mock_api_client():
    """Mock API client with health check and basic operations."""
    client = MagicMock()
    
    # Mock health check
    client.get_health.return_value = {"status": "healthy"}
    
    # Mock document operations
    client.get_documents.return_value = [
        {
            "id": "doc1",
            "filename": "test.pdf",
            "page_count": 5,
            "timestamp": "2025-07-27T10:00:00"
        }
    ]
    
    # Mock upload operation
    client.upload_document.return_value = True
    
    # Mock query operation
    client.query_documents.return_value = {
        "response": "Test response",
        "source_documents": [
            {
                "filename": "test.pdf",
                "page_count": 1
            }
        ]
    }
    
    return client

@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client."""
    client = MagicMock()
    
    # Mock collection operations
    collection = MagicMock()
    collection.count.return_value = 5
    client.get_or_create_collection.return_value = collection
    
    return client

@pytest.fixture
def test_pdf_path():
    """Create a minimal test PDF for testing."""
    return str(Path(__file__).parent / "data" / "test.pdf")
