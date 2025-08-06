# Use Python base image with CUDA support for RunPod
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    wget \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install llama.cpp - Build from source for reliability
RUN apt-get update && apt-get install -y git cmake libcurl4-openssl-dev && \
    git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    mkdir build && cd build && \
    cmake .. -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF && \
    cmake --build . --config Release -j$(nproc) && \
    cp bin/llama-cli /usr/local/bin/llama && \
    cd ../.. && rm -rf llama.cpp && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY handler.py .

# Create model directory
RUN mkdir -p model

# Set environment variables for RunPod
ENV PYTHONPATH=/app
ENV RUNPOD_AI_API_KEY=""

# The model will be downloaded on first run to avoid build timeouts
# Expose port (RunPod handles this, but good practice)
EXPOSE 8000

# RunPod will run this automatically
CMD ["python", "handler.py"]