FROM python:3.10-slim

# Install basic tools
RUN apt-get update && apt-get install -y \
    git build-essential cmake wget curl

# Install llama.cpp
WORKDIR /usr/local/src
RUN git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
RUN cp llama.cpp/main /usr/local/bin/llama

# Create app directory
WORKDIR /app
COPY ./app /app/app
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
