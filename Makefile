# Install dependencies 
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Run the main analysis script 
run:
	python Analysis.py

# Run complete workflow
all: install run

# Help command
help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies (includes polars)"
	@echo "  make run          - Run main analysis (includes pandas vs polars)"
	@echo "  make all          - Run complete workflow"
	@echo "  make help         - Show this help message"

.PHONY: install run all help