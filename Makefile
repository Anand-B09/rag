# Makefile for RAG Docker Project

# Variables
PYTHON = python3
PYTEST = pytest
PYTEST_FLAGS = -v
PYTEST_COV_FLAGS = --cov=frontend --cov=backend --cov-report=term-missing
LOCUST = locust
DOCKER_COMPOSE = docker compose  # Use 'docker compose' instead of 'docker-compose' depending on your Docker version

# Docker commands
.PHONY: build up down logs backend frontend
.PHONY: test-unit install-test clean

build:
	$(DOCKER_COMPOSE) build

up:
	$(DOCKER_COMPOSE) up -d

down:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs -f

backend:
	$(DOCKER_COMPOSE) up -d backend

frontend:
	$(DOCKER_COMPOSE) up -d frontend

# Test commands
install-dev:
	$(PYTHON) -m pip install -e ".[test,dev]"

install-test:
	$(PYTHON) -m pip install -e ".[test]"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

test-unit:
	$(PYTEST) $(PYTEST_FLAGS) tests/

# Help
help:
	@echo "RAG System Commands:"
	@echo "Docker Commands:"
	@echo "  make build             Build all containers"
	@echo "  make up                Start all containers"
	@echo "  make down              Stop all containers"
	@echo "  make logs              View container logs"
	@echo "  make backend           Start backend container"
	@echo "  make frontend          Start frontend container"
	@echo ""
	@echo "Test Commands:"
	@echo "  make install-test      Install test dependencies"
	@echo "  make test-unit         Run unit tests"
	@echo ""
	@echo "  make clean             Clean up cache files"
