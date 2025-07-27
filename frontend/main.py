import streamlit as st
import logging
from datetime import datetime
from functools import lru_cache
from typing import Optional, List, Any, Dict
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from api_client import APIClient
from models import DocumentMetadata, ChatMessage
from styles import STYLES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application Settings
class Settings(BaseSettings):
    """Frontend application settings."""
    backend_host: str = Field("http://backend:8001")
    max_retries: int = Field(3)
    retry_delay: float = Field(1.0)
    chat_history_limit: int = Field(50)
    connection_timeout: float = Field(10.0)
    
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()



# Initialize Streamlit page
st.set_page_config(
    page_title="Local PDF RAG",
    page_icon="📄🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styles
st.markdown(STYLES, unsafe_allow_html=True)


class RAGApp:
    def __init__(self):
        """Initialize the RAG application."""
        self.settings = get_settings()
        self.api_client = APIClient(self.settings)
        self._init_session_state()
        self._check_backend_health()

    def _init_session_state(self):
        """Initialize session state variables."""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "documents" not in st.session_state:
            st.session_state.documents = []
        if "backend_healthy" not in st.session_state:
            st.session_state.backend_healthy = False

    def _check_backend_health(self) -> None:
        """Check and update backend health status."""
        try:
            health = self.api_client.get_health()
            st.session_state.backend_healthy = (
                health is not None and 
                health["status"] == "healthy"
            )
            
            if not st.session_state.backend_healthy:
                st.error("⚠️ Backend service is not healthy")
                logger.error("Backend health check failed")
            else:
                logger.info("Backend health check passed")
        except Exception as e:
            st.session_state.backend_healthy = False
            st.error(f"⚠️ Backend service unavailable: {str(e)}")
            logger.error(f"Backend health check error: {str(e)}")

    def _upload_documents(self, files: List[Any]) -> None:
        """Upload and process documents."""
        if not st.session_state.backend_healthy:
            st.error("Cannot upload documents while backend is unhealthy")
            return
            
        for file in files:
            if not file.name.lower().endswith('.pdf'):
                st.error(f"Unsupported file type: {file.name}")
                continue
                
            with st.spinner(f"Processing {file.name}..."):
                try:
                    result = self.api_client.upload_document(file)
                    if result:
                        st.success(f"Successfully processed {file.name}")
                        self._refresh_documents()
                    else:
                        st.error(f"Failed to process {file.name}")
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {str(e)}")
                    st.error(f"Error processing {file.name}: {str(e)}")

    def _query_documents(self, query: str) -> None:
        """Query processed documents."""
        if not st.session_state.backend_healthy:
            st.error("Cannot query documents while backend is unhealthy")
            return
            
        if not query.strip():
            return
        
        with st.spinner("Thinking..."):
            try:
                response = self.api_client.query_documents(query)
                if response:
                    message = ChatMessage(
                        query=query,
                        response=response["response"],
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        source_documents=response.get("source_documents", [])
                    )
                    
                    # Update chat history with limit
                    st.session_state.chat_history.append(message)
                    if len(st.session_state.chat_history) > self.settings.chat_history_limit:
                        st.session_state.chat_history.pop(0)
                    
                    # Display the response
                    self._display_chat_message(message)
                    
            except Exception as e:
                logger.error(f"Query failed: {str(e)}")
                st.error(f"Query failed: {str(e)}")

    def _display_chat_message(self, message: ChatMessage) -> None:
        """Display a chat message with sources."""
        with st.container():
            st.markdown(f"**Q:** {message['query']}")
            st.markdown(f"**A:** {message['response']}")
            
            if message['source_documents']:
                with st.expander("View Sources"):
                    for doc in message['source_documents']:
                        st.markdown(
                            f"📄 {doc['filename']} "
                            f"(Page {doc.get('page_count', 'N/A')})"
                        )
            
            st.caption(f"Time: {message['timestamp']}")
            st.divider()

    def _refresh_documents(self) -> None:
        """Refresh the list of processed documents."""
        try:
            documents = self.api_client.get_documents()
            if documents is not None:
                st.session_state.documents = documents
        except Exception as e:
            logger.error(f"Error refreshing documents: {str(e)}")

    def _delete_document(self, doc_id: str) -> None:
        """Delete a document from the system."""
        if not st.session_state.backend_healthy:
            st.error("Cannot delete documents while backend is unhealthy")
            return
            
        try:
            if self.api_client.delete_document(doc_id):
                st.success("Document deleted successfully")
                self._refresh_documents()
            else:
                st.error("Failed to delete document")
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            st.error(f"Error deleting document: {str(e)}")

    def display_sidebar(self) -> None:
        """Display and handle sidebar elements."""
        with st.sidebar:
            st.markdown('<p class="sidebar-header">📚 Document Management</p>', unsafe_allow_html=True)
            
            # System status indicator
            if st.session_state.backend_healthy:
                st.markdown('<p class="status-success">✅ System Online</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-error">❌ System Offline</p>', unsafe_allow_html=True)
                return
            
            # Document upload section
            uploaded_files = st.file_uploader(
                "Upload PDF Documents",
                type="pdf",
                accept_multiple_files=True,
                help="Select one or more PDF files to upload"
            )
            
            if uploaded_files:
                self._upload_documents(uploaded_files)
            
            # Document list
            if st.session_state.documents:
                st.markdown("### Processed Documents")
                for doc in st.session_state.documents:
                    with st.container():
                        # Document card
                        st.markdown(
                            f"""
                            <div class="document-card">
                                <h4>📄 {doc['filename']}</h4>
                                <p class="status-info">Pages: {doc.get('page_count', 'N/A')}</p>
                                <p class="message-timestamp">Added: {doc['timestamp']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Delete button
                        if st.button("🗑️ Delete", key=f"delete_{doc['id']}"):
                            self._delete_document(doc['id'])
            else:
                st.info("No documents uploaded yet")

    def display_chat(self) -> None:
        """Display chat interface and handle queries."""
        st.markdown('<p class="main-header">Local PDF RAG Assistant</p>', unsafe_allow_html=True)
        
        if not st.session_state.backend_healthy:
            st.error("⚠️ System is currently offline. Please try again later.")
            return
        
        # Query input
        query = st.text_input(
            "Ask a question about your documents:",
            help="Enter your question here"
        )
        
        if query:
            self._query_documents(query)
        
        # Chat history
        if st.session_state.chat_history:
            st.markdown("### Chat History")
            
            # Allow clearing chat history
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
            
            # Display messages
            for item in reversed(st.session_state.chat_history):
                self._display_chat_message(item)

    def run(self) -> None:
        """Run the Streamlit application."""
        try:
            self.display_sidebar()
            self.display_chat()
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error("An unexpected error occurred. Please refresh the page.")

def main():
    """Initialize and run the application."""
    try:
        app = RAGApp()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        st.error("Failed to start the application. Please try again later.")

if __name__ == "__main__":
    main()