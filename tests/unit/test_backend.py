"""Unit tests for backend service components."""
import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_upload_endpoint(test_pdf_path):
    """Test document upload endpoint."""
    with open(test_pdf_path, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200
    assert response.json()["success"] == True

def test_query_endpoint():
    """Test the query endpoint."""
    response = client.post(
        "/query",
        json={"query": "test query"}
    )
    assert response.status_code == 200
    assert "response" in response.json()
    assert "source_documents" in response.json()

def test_error_handling():
    """Test error handling in endpoints."""
    # Test invalid file type
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"test", "text/plain")}
    )
    assert response.status_code == 400
    
    # Test missing query
    response = client.post("/query", json={})
    assert response.status_code == 422
