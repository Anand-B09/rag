# Makefile for RAG Docker Project

# Variables
PYTHON = python3
PYTEST = pytest
PYTEST_FLAGS = -v
PYTEST_COV_FLAGS = --cov=frontend --cov=backend --cov-report=term-missing
LOCUST = locust
DOCKER_COMPOSE = docker compose  # Use 'docker compose' instead of 'docker-compose'

# Docker commands
.PHONY: build up down logs backend frontend
.PHONY: test test-all test-unit test-integration test-performance test-docker install-test clean

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
	$(PYTEST) $(PYTEST_FLAGS) tests/unit/

test-integration:
	$(PYTEST) $(PYTEST_FLAGS) tests/integration/

test-performance:
	$(LOCUST) -f tests/performance/locustfile.py --host http://localhost:8000 \
		--headless -u 10 -r 1 --run-time 30s

test-performance-web:
	$(LOCUST) -f tests/performance/locustfile.py --host http://localhost:8000

test-coverage:
	$(PYTEST) $(PYTEST_FLAGS) $(PYTEST_COV_FLAGS) tests/

# Docker-based testing
test-frontend-docker:
	$(DOCKER_COMPOSE) run --rm frontend $(PYTEST) $(PYTEST_FLAGS) tests/unit/test_frontend.py

test-backend-docker:
	$(DOCKER_COMPOSE) run --rm backend $(PYTEST) $(PYTEST_FLAGS) tests/unit/test_backend.py

test-integration-docker:
	$(DOCKER_COMPOSE) run --rm backend $(PYTEST) $(PYTEST_FLAGS) tests/integration/

test-docker-all: test-frontend-docker test-backend-docker test-integration-docker

# Combined test commands
test-all: test-unit test-integration test-coverage

test-local: clean install-test test-all

# Development helpers
watch-tests:
	$(PYTEST) $(PYTEST_FLAGS) --looponfail tests/

coverage-report: test-coverage
	coverage html
	@echo "Coverage report generated in htmlcov/index.html"

# Help
help:
	@echo "RAG System Commands:"
	@echo "Docker Commands:"
	@echo "  make build              Build all containers"
	@echo "  make up                Start all containers"
	@echo "  make down              Stop all containers"
	@echo "  make logs              View container logs"
	@echo "  make backend           Start backend container"
	@echo "  make frontend          Start frontend container"
	@echo ""
	@echo "Test Commands:"
	@echo "  make install-test      Install test dependencies"
	@echo "  make test-unit         Run unit tests"
	@echo "  make test-integration  Run integration tests"
	@echo "  make test-performance  Run performance tests"
	@echo "  make test-coverage     Run tests with coverage"
	@echo "  make test-all          Run all tests"
	@echo "  make test-local        Run all tests locally"
	@echo "  make test-docker-all   Run all tests in Docker"
	@echo ""
	@echo "Development Commands:"
	@echo "  make watch-tests       Run tests in watch mode"
	@echo "  make coverage-report   Generate HTML coverage report"
	@echo "  make clean             Clean up cache files"
