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

# Run the main analysis script 
run:
	@echo "🚀 Running main analysis..."
	python Analysis.py

# Run complete workflow
all: install run

# Help command
help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies (includes polars)"
	@echo "  make run          - Run main analysis (includes pandas vs polars)"
	@echo "  make test        - Run all tests with verbose output and coverage"
	@echo "  make all          - Run complete workflow"
	@echo "  make help         - Show this help message"