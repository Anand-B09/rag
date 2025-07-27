
import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

# Configure Streamlit page
st.set_page_config(
    page_title="📄 RAG Document Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")

class RAGClient:
    """Client class to interact with the FastAPI backend"""

    def __init__(self, backend_url: str):
        self.backend_url = backend_url.rstrip('/')

    def health_check(self) -> Dict[str, Any]:
        """Check backend health status"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"status": "unhealthy", "error": str(e)}

    def upload_documents(self, files: List[tuple]) -> Dict[str, Any]:
        """Upload PDF documents to backend"""
        try:
            file_data = []
            for file_name, file_content in files:
                file_data.append(('files', (file_name, file_content, 'application/pdf')))

            response = requests.post(
                f"{self.backend_url}/ingest",
                files=file_data,
                timeout=300
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

    def query_documents(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the document store"""
        try:
            payload = {
                "query": query,
                "top_k": top_k,
                "enable_streaming": False
            }

            response = requests.post(
                f"{self.backend_url}/query",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

# Initialize client
@st.cache_resource
def get_rag_client():
    return RAGClient(BACKEND_URL)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_uploaded" not in st.session_state:
        st.session_state.documents_uploaded = False

def display_system_status():
    """Display system health and status in sidebar"""
    st.sidebar.header("🔧 System Status")

    client = get_rag_client()
    health_data = client.health_check()

    if health_data.get("status") == "healthy":
        st.sidebar.success("✅ System Healthy")
        st.sidebar.metric("Documents Indexed", health_data.get("documents_count", 0))
        st.sidebar.metric("ChromaDB", "Connected" if health_data.get("chroma_connected") else "Disconnected")
    else:
        st.sidebar.error("❌ System Unhealthy")
        if "error" in health_data:
            st.sidebar.error(f"Error: {health_data['error']}")

def document_upload_section():
    """Handle document upload"""
    st.header("📤 Upload PDF Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files to upload and index",
        type=['pdf'],
        accept_multiple_files=True,
        help="Select one or more PDF documents. Each document will be processed page by page."
    )

    if uploaded_files:
        st.info(f"📁 Selected {len(uploaded_files)} file(s)")

        if st.button("🚀 Upload and Process Documents", type="primary"):
            if len(uploaded_files) > 10:
                st.error("⚠️ Please upload no more than 10 files at once")
                return

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Prepare files for upload
                files_data = []
                total_size = 0

                for i, uploaded_file in enumerate(uploaded_files):
                    file_content = uploaded_file.read()
                    file_size = len(file_content)

                    if file_size > 50 * 1024 * 1024:  # 50MB limit
                        st.error(f"❌ File '{uploaded_file.name}' exceeds 50MB limit")
                        return

                    total_size += file_size
                    files_data.append((uploaded_file.name, file_content))

                    progress_bar.progress((i + 1) / len(uploaded_files) * 0.5)
                    status_text.text(f"Preparing {uploaded_file.name}...")

                if total_size > 200 * 1024 * 1024:  # 200MB total limit
                    st.error("❌ Total file size exceeds 200MB limit")
                    return

                status_text.text("🔄 Uploading and processing documents...")

                client = get_rag_client()
                result = client.upload_documents(files_data)

                progress_bar.progress(1.0)

                if result.get("success"):
                    data = result["data"]
                    st.success(f"✅ Successfully processed {data['documents_processed']} documents in {data['processing_time']:.2f} seconds")
                    st.session_state.documents_uploaded = True

                    # Show summary
                    with st.expander("📊 Processing Summary", expanded=True):
                        st.write(f"**Message:** {data['message']}")
                        st.write(f"**Documents Processed:** {data['documents_processed']}")
                        st.write(f"**Processing Time:** {data['processing_time']:.2f} seconds")

                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ Upload failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"❌ Error during upload: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()

def chat_interface():
    """Main chat interface for querying documents"""
    st.header("💬 Chat with Your Documents")

    # Check if documents are available
    client = get_rag_client()
    health_data = client.health_check()

    if health_data.get("documents_count", 0) == 0:
        st.warning("⚠️ No documents uploaded yet. Please upload some PDF documents first!")
        return

    # Settings in sidebar
    st.sidebar.header("⚙️ Chat Settings")
    top_k = st.sidebar.slider("Number of sources to retrieve", min_value=1, max_value=10, value=5)

    # Clear chat history button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Display current document count
    st.sidebar.info(f"📚 {health_data.get('documents_count', 0)} documents available")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])

                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            **Source {i}: {source.get('source', 'Unknown')}** (Page {source.get('page', 'Unknown')})
                            - **Relevance Score:** {source.get('score', 0):.3f}
                            - **Text Preview:** *{source.get('text', '')[:200]}...*
                            """)

                # Display processing time
                if "processing_time" in message:
                    st.caption(f"⏱️ Response time: {message['processing_time']:.2f}s")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching documents and generating response..."):
                result = client.query_documents(query=prompt, top_k=top_k)

                if result.get("success"):
                    data = result["data"]
                    response_text = data.get("response", "No response generated")
                    sources = data.get("sources", [])
                    processing_time = data.get("processing_time", 0)

                    st.markdown(response_text)

                    # Display sources
                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"""
                                **Source {i}: {source.get('source', 'Unknown')}** (Page {source.get('page', 'Unknown')})
                                - **Relevance Score:** {source.get('score', 0):.3f}
                                - **Text Preview:** *{source.get('text', '')[:200]}...*
                                """)

                    # Display processing time
                    st.caption(f"⏱️ Response time: {processing_time:.2f}s")

                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources,
                        "processing_time": processing_time
                    })
                else:
                    error_msg = f"❌ Query failed: {result.get('error', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

def main():
    """Main application function"""
    initialize_session_state()

    # Header
    st.title("📄🤖 RAG Document Assistant")
    st.markdown("Upload PDF documents and chat with them using AI-powered search and generation.")

    # System status in sidebar
    display_system_status()

    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    with col1:
        document_upload_section()

    with col2:
        chat_interface()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**RAG Assistant v1.0**")
    st.sidebar.markdown("Built with Streamlit & FastAPI")
    st.sidebar.markdown("Vector DB: ChromaDB | LLM: Ollama")

if __name__ == "__main__":
    main()
