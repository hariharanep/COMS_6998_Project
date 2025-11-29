from __future__ import annotations

import os
import socket
from datetime import datetime

from typing import Dict, List
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi import Query, Path
from typing import Optional

from models.health import Health
from models.prompt import EnhancePromptReq, GeneratePromptReq

import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1

# Load GPT-2
t5_pipe = pipeline("text-generation", model="google/flan-t5-base", device=device)
gpt2_pipe = pipeline("text-generation", model="gpt2", device=device)

model_to_pipe = {
    "gpt2": gpt2_pipe,
    "google/flan-t5-base": t5_pipe
}

port = int(os.environ.get("FASTAPIPORT", 8000))

app = FastAPI(
    title="Backend Service API",
    version="0.0.1",
)

# -----------------------------------------------------------------------------
# Address endpoints
# -----------------------------------------------------------------------------

def make_health(echo: Optional[str], path_echo: Optional[str]=None) -> Health:
    return Health(
        status=200,
        status_message="OK",
        timestamp=datetime.utcnow().isoformat() + "Z",
        ip_address=socket.gethostbyname(socket.gethostname()),
        echo=echo,
        path_echo=path_echo
    )

@app.get("/health", response_model=Health)
def get_health_no_path(echo: str | None = Query(None, description="Optional echo string")):
    # Works because path_echo is optional in the model
    return make_health(echo=echo, path_echo=None)

@app.get("/health/{path_echo}", response_model=Health)
def get_health_with_path(
    path_echo: str = Path(..., description="Required echo in the URL path"),
    echo: str | None = Query(None, description="Optional echo string"),
):
    return make_health(echo=echo, path_echo=path_echo)

@app.post("/enhance")
def enhance(req: EnhancePromptReq):
    pipeline = model_to_pipe[req.model]
    enhanced = pipeline(req.prompt, max_length=256)[0]['generated_text']
    return {"enhanced_prompt": enhanced}

@app.post("/generate")
def generate(req: GeneratePromptReq):
    pipeline = model_to_pipe[req.model]
    out = pipeline(req.prompt, max_length=256)[0]["generated_text"]
    return {"enhanced_response": out}

# -----------------------------------------------------------------------------
# Root
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Backend Service API."}

# -----------------------------------------------------------------------------
# Entrypoint for `python main.py`
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
