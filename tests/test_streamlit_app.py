import pytest
import builtins
from unittest.mock import patch, Mock, MagicMock
import streamlit as st

import frontend.streamlit_app as streamlit_app


@pytest.fixture
def mock_requests_post():
    with patch("streamlit_app.requests.post") as mock_post:
        yield mock_post


@pytest.fixture
def mock_requests_get():
    with patch("streamlit_app.requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def mock_st_file_uploader():
    with patch("streamlit.file_uploader") as mock_uploader:
        yield mock_uploader


def test_health_check_success(mock_requests_get):
    # Mock successful GET response
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "healthy",
        "chroma_connected": True,
        "documents_count": 5,
    }
    mock_requests_get.return_value = mock_resp

    client = streamlit_app.RAGClient("http://example.com")
    health = client.health_check()

    assert health["status"] == "healthy"
    assert health["chroma_connected"] is True
    assert health["documents_count"] == 5


def test_health_check_failure(mock_requests_get):
    # Simulate request exception
    mock_requests_get.side_effect = Exception("Connection error")

    client = streamlit_app.RAGClient("http://example.com")
    health = client.health_check()

    assert health["status"] == "unhealthy"
    assert "Connection error" in health["error"]


def test_upload_documents_success(mock_requests_post):
    # Mock successful POST response for uploading documents
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"documents_processed": 1, "message": "Success", "processing_time": 1.23}
    mock_requests_post.return_value = mock_resp

    client = streamlit_app.RAGClient("http://example.com")

    files = [("file1.pdf", b"filecontent")]
    result = client.upload_documents(files)

    assert result["success"] is True
    assert "documents_processed" in result["data"]


def test_upload_documents_failure(mock_requests_post):
    mock_resp = Mock()
    mock_resp.status_code = 400
    mock_resp.text = "Bad Request"
    mock_requests_post.return_value = mock_resp

    client = streamlit_app.RAGClient("http://example.com")

    files = [("file1.pdf", b"filecontent")]
    result = client.upload_documents(files)

    assert result["success"] is False
    assert "HTTP 400" in result["error"]


def test_query_documents_success(mock_requests_post):
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "Here is your answer",
        "sources": [],
        "processing_time": 0.55,
    }
    mock_requests_post.return_value = mock_resp

    client = streamlit_app.RAGClient("http://example.com")

    result = client.query_documents("test query", top_k=5)

    assert result["success"] is True
    assert "response" in result["data"]


def test_query_documents_failure(mock_requests_post):
    mock_resp = Mock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_requests_post.return_value = mock_resp

    client = streamlit_app.RAGClient("http://example.com")

    result = client.query_documents("test query")

    assert result["success"] is False
    assert "HTTP 500" in result["error"]


@patch("streamlit_app.st.file_uploader")
@patch("streamlit_app.st.file_uploader")
@patch("streamlit_app.st.button")
@patch("streamlit_app.st.progress")
@patch("streamlit_app.st.empty")
def test_document_upload_section_upload_and_process_documents(
    mock_empty, mock_progress, mock_button, mock_file_uploader
):
    # Setup mock for file_uploader to return two fake files
    class FakeUploadedFile:
        def __init__(self, name, content):
            self.name = name
            self._content = content
        def read(self):
            return self._content

    fake_files = [
        FakeUploadedFile("doc1.pdf", b"pdfcontent1"),
        FakeUploadedFile("doc2.pdf", b"pdfcontent2"),
    ]
    mock_file_uploader.return_value = fake_files

    # Mock button returns True to simulate clicking "Upload and Process"
    mock_button.return_value = True

    # Mock progress and empty returning mocks
    mock_progress.return_value = MagicMock()
    mock_empty.return_value = MagicMock()

    # Mock the RAGClient.upload_documents method call
    with patch.object(streamlit_app.RAGClient, "upload_documents") as mock_upload_documents:
        mock_upload_documents.return_value = {
            "success": True,
            "data": {
                "documents_processed": 2,
                "message": "Processed successfully",
                "processing_time": 3.14,
            },
        }

        # Call the document upload section function
        streamlit_app.document_upload_section()

        # Assert upload_documents was called once
        mock_upload_documents.assert_called_once()

@patch("streamlit_app.st.chat_input")
@patch("streamlit_app.st.sidebar.slider")
@patch("streamlit_app.st.sidebar.button")
@patch("streamlit_app.st.session_state", new_callable=lambda: {"messages": []})
@patch("streamlit_app.st.chat_message")
@patch("streamlit_app.st.sidebar.info")
def test_chat_interface_basic_flow(mock_sidebar_info, mock_chat_message, mock_sidebar_button, mock_sidebar_slider, mock_chat_input, mock_session_state):
    # Setup mocks
    mock_sidebar_slider.return_value = 5
    mock_sidebar_button.return_value = False  # Clear chat history button not clicked
    mock_chat_input.return_value = None        # No new input

    # Mock RAGClient.health_check to simulate having documents
    with patch.object(streamlit_app.RAGClient, "health_check") as mock_health_check:
        mock_health_check.return_value = {"documents_count": 1}

        streamlit_app.chat_interface()

        mock_sidebar_info.assert_called()
        mock_health_check.assert_called_once()


