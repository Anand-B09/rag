# 1. Health check
curl -s http://localhost:8000/api/v1/heartbeat

# 2. List initial collections (should be empty or default)
curl -s http://localhost:8000/api/v1/collections

# 3. Create a new collection
curl -s -X POST http://localhost:8000/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "unit_test_collection"}'

# 4. List collections to verify creation
curl -s http://localhost:8000/api/v1/collections

# 5. Add a test document to the collection
curl -s -X POST http://localhost:8000/api/v1/collections/unit_test_collection/add \
  -H "Content-Type: application/json" \
  -d '{"ids": ["doc1"], "metadatas": [{"source": "unit_test"}], "documents": ["Hello Chroma!"], "embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]}'

# 6. Query the collection for the document
curl -s -X POST http://localhost:8000/api/v1/collections/unit_test_collection/query \
  -H "Content-Type: application/json" \
  -d '{"query_embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], "n_results": 1}'

# 7. Clean up: Delete the test collection
curl -s -X DELETE http://localhost:8000/api/v1/collections/unit_test_collection

# 8. Final check: List collections again to verify deletion
curl -s http://localhost:8000/api/v1/collections
