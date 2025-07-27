"""Integration tests for complete RAG system flow."""
import pytest
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from backend.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

def create_test_pdf(size_kb: int = 10) -> bytes:
    """Create a test PDF of specified size."""
    # Basic PDF structure
    pdf_header = b"%PDF-1.4\n"
    pdf_footer = b"%%EOF\n"
    
    # Create content to match size
    content_size = size_kb * 1024 - len(pdf_header) - len(pdf_footer)
    content = b"0" * content_size
    
    return pdf_header + content + pdf_footer

@pytest.mark.asyncio
async def test_complete_rag_flow():
    """Test complete RAG flow from upload to query."""
    # Create and upload test document
    test_pdf = create_test_pdf()
    upload_response = client.post(
        "/upload",
        files={"file": ("test.pdf", test_pdf, "application/pdf")}
    )
    assert upload_response.status_code == 200
    doc_id = upload_response.json()["id"]
    
    # Wait for processing
    time.sleep(2)
    
    # Query the uploaded content
    query_response = client.post(
        "/query",
        json={"query": "What is in the document?"}
    )
    assert query_response.status_code == 200
    
    # Verify response
    response_data = query_response.json()
    assert "response" in response_data
    assert "source_documents" in response_data
    
    # Clean up
    delete_response = client.delete(f"/documents/{doc_id}")
    assert delete_response.status_code == 200

@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test system under concurrent operations."""
    def upload_and_query():
        # Upload
        pdf = create_test_pdf(5)  # 5KB test PDF
        upload_resp = client.post(
            "/upload",
            files={"file": ("test.pdf", pdf, "application/pdf")}
        )
        assert upload_resp.status_code in [200, 429]
        
        if upload_resp.status_code == 200:
            # Query
            query_resp = client.post(
                "/query",
                json={"query": "Test query"}
            )
            assert query_resp.status_code in [200, 429]
            
            # Delete if upload succeeded
            doc_id = upload_resp.json().get("id")
            if doc_id:
                delete_resp = client.delete(f"/documents/{doc_id}")
                assert delete_resp.status_code in [200, 404]
        
        return upload_resp.status_code
    
    # Run concurrent operations
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(upload_and_query) for _ in range(10)]
        results = [f.result() for f in futures]
        
        # Verify that at least some operations succeeded
        assert any(status == 200 for status in results)

@pytest.mark.asyncio
async def test_error_recovery():
    """Test system recovery from errors."""
    # Test with malformed PDF
    bad_pdf = b"Not a PDF file"
    response = client.post(
        "/upload",
        files={"file": ("bad.pdf", bad_pdf, "application/pdf")}
    )
    assert response.status_code == 400
    
    # Test with valid PDF after error
    good_pdf = create_test_pdf()
    response = client.post(
        "/upload",
        files={"file": ("good.pdf", good_pdf, "application/pdf")}
    )
    assert response.status_code == 200
    
    # Verify system still healthy
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_large_document_handling():
    """Test system with large documents."""
    sizes = [100, 500, 1000]  # KB sizes
    
    for size in sizes:
        pdf = create_test_pdf(size)
        response = client.post(
            "/upload",
            files={"file": (f"large_{size}kb.pdf", pdf, "application/pdf")}
        )
        
        if size > 500:  # Assuming 500KB limit
            assert response.status_code == 413  # Payload too large
        else:
            assert response.status_code == 200
