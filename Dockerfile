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

# Set work directory
WORKDIR /app

# Copy all app files
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r Requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Run the app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
