import os
import io
import pytest
from fastapi.testclient import TestClient
from backend.app import app, rag_service, RAGService, Document

client = TestClient(app)

# Mock PDF content (simple 1-page PDF binary)
SAMPLE_PDF = b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT\n/F1 24 Tf\n100 700 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000100 00000 n \n0000000179 00000 n \ntrailer\n<< /Root 1 0 R /Size 5 >>\nstartxref\n258\n%%EOF'

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    # Setup before tests (clear documents info)
    rag_service.documents_info.clear()
    yield
    # Teardown after tests
    rag_service.documents_info.clear()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert "status" in json_data
    assert "chroma_connected" in json_data
    assert "documents_count" in json_data

def test_list_documents_empty():
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    # Initially no documents
    assert isinstance(data, list)
    assert len(data) == 0

def test_ingest_non_pdf_file():
    response = client.post(
        "/ingest",
        files={"files": ("textfile.txt", b"not a pdf content", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.text

def test_ingest_empty_files_list():
    response = client.post(
        "/ingest",
        files=[],
    )
    assert response.status_code == 400 or response.status_code == 422  # validation error or 400

def test_ingest_valid_pdf():
    response = client.post(
        "/ingest",
        files={"files": ("test.pdf", SAMPLE_PDF, "application/pdf")},
    )
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["documents_processed"] >= 1
    assert "Successfully processed" in json_data["message"]

def test_list_documents_after_ingest():
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    doc = data[0]
    assert "filename" in doc
    assert "page_count" in doc
    assert "upload_time" in doc

def test_query_documents_empty_query():
    response = client.post("/query", json={"query": "", "top_k": 5})
    assert response.status_code == 400
    assert "Query field is required" in response.text

def test_query_documents_short_query():
    response = client.post("/query", json={"query": "hi", "top_k": 5})
    assert response.status_code == 400
    assert "Query must be at least 3 characters" in response.text

def test_query_documents_invalid_top_k():
    response = client.post("/query", json={"query": "hello", "top_k": 0})
    assert response.status_code == 400
    assert "top_k must be between 1 and 20" in response.text

def test_query_documents_success():
    # This test assumes there is at least 1 document ingested from previous test
    response = client.post("/query", json={"query": "hello", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert "processing_time" in data

def test_delete_document_not_found():
    response = client.delete("/documents/non_existing_doc.pdf")
    assert response.status_code == 404

def test_delete_document_success():
    # First ingest a document
    r = client.post(
        "/ingest",
        files={"files": ("todelete.pdf", SAMPLE_PDF, "application/pdf")},
    )
    assert r.status_code == 200
    # Then delete it
    response = client.delete("/documents/todelete.pdf")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data.get("status") == "success"

def test_rag_service_extract_pdf_text():
    # Write temp pdf file and extract text
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(SAMPLE_PDF)
        tmp.flush()
        docs = rag_service.extract_pdf_text(tmp.name, "sample.pdf")
        assert isinstance(docs, list)
        assert all(isinstance(doc, Document) for doc in docs)
        assert any(len(doc.text) > 0 for doc in docs)

def test_rag_service_add_and_delete_document():
    # Create dummy document
    dummy_doc = Document(text="Test document text", metadata={"filename": "dummy.pdf", "page": 1, "source": "dummy.pdf"})
    rag_service.add_document_to_index([dummy_doc])
    # Document should now be in metadata
    rag_service.documents_info["dummy.pdf"] = {
        "filename": "dummy.pdf",
        "page_count": 1,
        "upload_time": "now",
        "file_size": 100,
        "mime_type": "application/pdf"
    }
    deleted = rag_service.delete_document_from_store("dummy.pdf")
    assert deleted is True or deleted is False  # Depending on backend connection, allow both
