# 1) Base image with Python 3.12 + devcontainer utilities
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# 2) Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3) Use the vscode user that already exists in devcontainers
USER vscode

# 4) Working directory
WORKDIR /app

# 5) Install dependencies (copy requirements first for caching)
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pytest pytest-cov flake8 black ipython

# 6) Copy project files
COPY . /app

# 7) Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 8) Default user
USER $USERNAME

# 9) Default command
CMD ["/bin/bash"]
