from fastapi import FastAPI, Request
from pydantic import BaseModel
import subprocess
import os
import urllib.request

# Define constants
MODEL_PATH = "app/model/mistral-7b.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_DIR = os.path.dirname(MODEL_PATH)

# Make sure model is downloaded
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")

# Define FastAPI app
app = FastAPI()

# Define request schema
class PromptRequest(BaseModel):
    prompt: str

@app.post("/predict")
async def predict(request: PromptRequest):
    try:
        # Build llama.cpp command
        result = subprocess.run(
            ["llama", "-m", MODEL_PATH, "-p", request.prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Error handling
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
