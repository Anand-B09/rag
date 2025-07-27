"""Unit tests for the ingest module."""
import pytest
from pathlib import Path
from frontend.ingest import (
    validate_pdf,
    pdf_to_docs,
    process_batch,
    process_file
)

def test_validate_pdf_success(test_pdf_path):
    """Test PDF validation with valid file."""
    is_valid, error = validate_pdf(test_pdf_path)
    assert is_valid
    assert error == ""

def test_validate_pdf_nonexistent():
    """Test PDF validation with nonexistent file."""
    is_valid, error = validate_pdf("nonexistent.pdf")
    assert not is_valid
    assert "does not exist" in error

def test_pdf_to_docs(test_pdf_path):
    """Test PDF to document conversion."""
    docs = list(pdf_to_docs(test_pdf_path))
    assert len(docs) > 0
    assert all(doc.text.strip() for doc in docs)

def test_process_file_success(test_pdf_path):
    """Test successful file processing."""
    docs, success = process_file(test_pdf_path)
    assert success
    assert len(docs) > 0

def test_process_file_failure():
    """Test file processing failure."""
    docs, success = process_file("nonexistent.pdf")
    assert not success
    assert len(docs) == 0
