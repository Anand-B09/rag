"""Test ChromaDB integration."""
import pytest
from unittest.mock import patch
import chromadb
from llama_index.core import Document

def test_chroma_connection(mock_chroma_client):
    """Test ChromaDB connection and basic operations."""
    collection = mock_chroma_client.get_or_create_collection("test_collection")
    assert collection is not None

def test_embedding_persistence():
    """Test that embeddings are correctly stored and retrieved."""
    client = chromadb.Client()
    collection = client.create_collection("test_embeddings")
    
    # Add test document
    doc = Document(text="Test document")
    collection.add(
        documents=[doc.text],
        metadatas=[{"source": "test"}],
        ids=["1"]
    )
    
    # Verify retrieval
    results = collection.get(ids=["1"])
    assert len(results["documents"]) == 1
    assert results["documents"][0] == "Test document"

def test_query_relevance():
    """Test query results relevance."""
    client = chromadb.Client()
    collection = client.create_collection("test_query")
    
    # Add test documents
    docs = [
        "The cat sat on the mat",
        "The dog played in the yard",
        "Python is a programming language"
    ]
    
    collection.add(
        documents=docs,
        ids=["1", "2", "3"]
    )
    
    # Test query relevance
    results = collection.query(
        query_texts=["What about cats?"],
        n_results=1
    )
    
    assert "cat" in results["documents"][0][0].lower()
