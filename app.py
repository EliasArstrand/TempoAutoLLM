from fastapi import FastAPI, Request
from pydantic import BaseModel
import subprocess
import os

app = FastAPI()

MODEL_PATH = "app/model/mistral-7b.gguf"  # Replace with your GGUF filename
LLAMA_BINARY = "/usr/local/bin/llama"     # Installed in Dockerfile

class PromptInput(BaseModel):
    prompt: str

@app.post("/predict")
async def predict(data: PromptInput):
    try:
        result = subprocess.check_output([
            LLAMA_BINARY,
            "-m", MODEL_PATH,
            "-p", data.prompt,
            "--n-predict", "100",
            "--temp", "0.7"
        ], stderr=subprocess.STDOUT)
        return {"output": result.decode("utf-8")}
    except subprocess.CalledProcessError as e:
        return {"error": e.output.decode("utf-8")}
