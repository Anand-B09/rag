"""Custom styles for the Streamlit frontend."""

STYLES = """
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
    text-align: center;
    color: #1f77b4;
}

.sidebar-header {
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
    color: #2c3e50;
}

.document-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    background-color: #f8f9fa;
}

.chat-message {
    margin-bottom: 1rem;
    padding: 1rem;
    border-radius: 8px;
}

.user-message {
    background-color: #e9ecef;
}

.assistant-message {
    background-color: #f8f9fa;
}

.message-timestamp {
    color: #6c757d;
    font-size: 0.8rem;
}

.status-success {
    color: #28a745;
    font-weight: bold;
}

.status-error {
    color: #dc3545;
    font-weight: bold;
}

.status-info {
    color: #17a2b8;
    font-weight: bold;
}

.source-documents {
    font-size: 0.9rem;
    color: #6c757d;
    margin-top: 0.5rem;
}
</style>
"""
