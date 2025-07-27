"""Performance and load testing for the RAG system."""
from locust import HttpUser, task, between
from locust.exception import StopUser
import random
import json

class RAGUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Check system health before starting load test."""
        response = self.client.get("/health")
        if response.status_code != 200:
            raise StopUser("System unhealthy")
    
    @task(3)  # Higher weight for queries
    def query_documents(self):
        """Test document querying under load."""
        queries = [
            "What are the main concepts?",
            "Can you summarize this?",
            "What are the key findings?",
            "Explain the methodology",
            "What are the conclusions?"
        ]
        query = random.choice(queries)
        
        with self.client.post(
            "/query",
            json={"query": query},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                if "response" in response.json():
                    response.success()
                else:
                    response.failure("No response in payload")
            elif response.status_code == 429:
                response.success()  # Rate limiting is expected behavior
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Regular health checks during load test."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                if response.json()["status"] == "healthy":
                    response.success()
                else:
                    response.failure("System unhealthy")
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def upload_document(self):
        """Test document uploads under load."""
        # Generate small test PDF content
        pdf_content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000015 00000 n\n0000000061 00000 n\n0000000114 00000 n\ntrailer\n<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF\n"
        
        with self.client.post(
            "/upload",
            files={"file": ("test.pdf", pdf_content, "application/pdf")},
            catch_response=True
        ) as response:
            if response.status_code in [200, 429]:  # Both success and rate limit are ok
                response.success()
            else:
                response.failure(f"Upload failed: {response.status_code}")
                
class BulkQueryUser(HttpUser):
    """User class for testing bulk operations and edge cases."""
    wait_time = between(5, 10)
    
    @task
    def bulk_query(self):
        """Test system with multiple queries in quick succession."""
        queries = [f"Query batch {i}" for i in range(5)]
        
        for query in queries:
            with self.client.post(
                "/query",
                json={"query": query},
                catch_response=True
            ) as response:
                if response.status_code not in [200, 429]:
                    response.failure(f"Bulk query failed: {response.status_code}")
                    break
