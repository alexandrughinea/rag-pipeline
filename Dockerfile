# Base image
FROM python:3.10-slim

# Install system dependencies based on architecture
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# For Apple Silicon (M1/M2)
FROM --platform=linux/arm64 python:3.10-slim AS arm64
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# For Intel/AMD
FROM --platform=linux/amd64 python:3.10-slim AS amd64
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Detect architecture and use appropriate base
FROM ${TARGETARCH:-amd64}

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Special handling for ctransformers based on architecture
RUN if [ "$(uname -m)" = "aarch64" ]; then \
        pip install --no-cache-dir ctransformers[cuda]; \
    else \
        pip install --no-cache-dir ctransformers; \
    fi

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
# Expose port
EXPOSE 8000
# Environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "src/main.py"]
