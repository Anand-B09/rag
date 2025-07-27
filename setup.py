"""Setup file for the project."""
from setuptools import setup, find_packages

setup(
    name="rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Frontend dependencies
        "streamlit>=1.31.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.6.0",
        "pydantic-settings>=2.1.0",
        
        # Backend dependencies
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "python-multipart>=0.0.6",
        "chromadb>=0.4.24",
        "llama-index-core>=0.10.0",
        "llama-index-llms-ollama>=0.1.1",
    ],
    extras_require={
        "test": [
            "pytest>=7.4.0",
            "pytest-mock>=3.11.1",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.23.0",
            "requests-mock>=1.11.0",
        ],
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "pylint>=2.17.0",
        ],
    }
)
