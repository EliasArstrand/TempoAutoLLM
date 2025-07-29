# Use Python base image
FROM python:3.10-slim

# Install dependencies for llama.cpp and Python
RUN apt-get update && \
    apt-get install -y wget unzip && \
    apt-get clean

# Install llama.cpp binary (precompiled)
RUN wget https://huggingface.co/ggerganov/llama.cpp/resolve/main/bin/llama-linux && \
    chmod +x llama-linux && \
    mv llama-linux /usr/local/bin/llama

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
