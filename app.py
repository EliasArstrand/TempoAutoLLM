from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import os
import urllib.request

# Model download settings
MODEL_PATH = "model/mistral-7b.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Download the model if it doesn't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading Mistral model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded.")

# Init FastAPI
app = FastAPI()

# Request schema
class PromptRequest(BaseModel):
    prompt: str

# POST endpoint
@app.post("/predict")
async def predict(request: PromptRequest):
    try:
        # Run the LLM using llama.cpp
        result = subprocess.run(
            ["llama", "-m", MODEL_PATH, "-p", request.prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            return {
                "error": result.stderr.strip(),
                "status": "llama.cpp failed"
            }

        return {
            "response": result.stdout.strip(),
            "status": "ok"
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "exception"
        }
