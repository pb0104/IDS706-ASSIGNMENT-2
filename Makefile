IMAGE_NAME=analysis-dev
CONTAINER_NAME=analysis-container
SCRIPT=Analysis.py

# Build the Docker image
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME):dev .

# Run container interactively
run-container:
	@echo "🚀 Running interactive container..."
	docker run --rm -it \
		-v $(PWD):/app \
		-p 8888:8888 \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) /bin/bash

# Stop the running container
stop:
	docker stop $(CONTAINER_NAME)

# Remove Docker Image
clean:
	docker rmi -f $(IMAGE_NAME)

# Install dependencies
install:
	@echo "📦 Installing required packages..."
	pip install -r requirements.txt
	pip install pytest pytest-cov

# Run tests with coverage
test:
	@echo "📊 Running all tests with coverage..."
	python -m pytest Test_Analysis.py \
	--cov=Analysis \
	--cov-report=term-missing

# Run the main analysis script locally
run:
	@echo "🚀 Running main analysis..."
	python $(SCRIPT)

# Run the main analysis script inside Docker
run-docker:
	@echo "🚀 Running main analysis inside Docker..."
	docker run --rm -it \
		-v $(PWD):/app \
		--entrypoint python \
		$(IMAGE_NAME) $(SCRIPT)

#Build and run container script in one step
up: 
	@echo "🚀 Building and running container..."
	build run-docker

# Run complete workflow
all: install run

# Format code using black
format:
	@echo "🎨 Formatting code with black..."
	python -m black *.py

# Lint code using flake8
lint:
	@echo "🔍 Linting code with flake8..."
	python -m flake8 *.py --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Format and lint together
format-lint: format lint
	@echo "✅ Code formatting and linting complete"
	
# Continuous Integration target
ci: install test
	@echo "✅ CI pipeline complete"


# Help command
help:
	@echo "Available commands:"
	@echo "  build           - Build the Docker image"
	@echo "  run-container   - Run an interactive container"
	@echo "  up              - Build and run the container"
	@echo "  stop            - Stop the running container"
	@echo "  clean           - Remove the Docker image"
	@echo "  make install      - Install dependencies (includes polars)"
	@echo "  make run          - Run main analysis locally"
	@echo "  make run-docker   - Run main analysis inside Docker"
	@echo "  make test        - Run all tests with verbose output and coverage"
	@echo "  make all          - Run complete workflow"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Lint code with flake8"
	@echo "  make format-lint  - Format and lint code"
	@echo "  make ci           - Run CI pipeline"
	@echo "  make help         - Show this help message"