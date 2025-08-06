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

# Install llama.cpp binary (CPU version for efficiency)
RUN wget -O llama-cpp.tar.gz "https://github.com/ggerganov/llama.cpp/releases/download/b3559/llama-b3559-bin-ubuntu-x64.zip" && \
    apt-get update && apt-get install -y unzip && \
    unzip llama-b3559-bin-ubuntu-x64.zip && \
    mv llama-b3559-bin-ubuntu-x64/llama-cli /usr/local/bin/llama && \
    chmod +x /usr/local/bin/llama && \
    rm -rf llama-cpp.tar.gz llama-b3559-bin-ubuntu-x64* && \
    apt-get clean

# Alternative: Build from source if binary doesn't work
# RUN git clone https://github.com/ggerganov/llama.cpp.git && \
#     cd llama.cpp && \
#     make -j$(nproc) && \
#     cp llama-cli /usr/local/bin/llama && \
#     cd .. && rm -rf llama.cpp

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