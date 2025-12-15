import os, base64, binascii
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openai import OpenAI

# Configuración
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

MODEL = os.getenv("MODEL", "gpt-4.1-mini")
MAX_REQ_BYTES = 32 * 1024 * 1024  # 32 MiB
client = OpenAI()

app = FastAPI(title="GPT Proxy", version="3.1-outputstring")

# Modelos Pydantic
class InferenceIn(BaseModel):
    text: str

# Endpoint de salud
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

# Endpoint de inferencia
@app.post("/infer")
def infer(payload: InferenceIn):
    try:
        # Procesar el texto y llamar al modelo
        response = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": [{"type": "input_text", "text": payload.text}]}],
        )

        # Devolver solo el texto generado
        return {"output": response.output_text}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

# Servir el archivo HTML
@app.get("/")
def frontend():
    return FileResponse("static/index.html")
