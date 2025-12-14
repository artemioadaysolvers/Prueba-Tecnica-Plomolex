import os, base64, binascii
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

# --- Configuración ---
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno.")

MODEL = os.getenv("MODEL", "gpt-4.1-mini")  # modelo con visión
MAX_REQ_BYTES = 32 * 1024 * 1024  # 32 MiB
client = OpenAI()

app = FastAPI(title="GPT Proxy", version="3.1-outputstring")

# --- Modelos Pydantic ---
class ImageInput(BaseModel):
    image_b64: str = Field(..., description="Imagen en base64 (sin prefijo data:)")
    mime: Optional[str] = Field(None, description="MIME type, ej: image/jpeg, image/png")

class InferenceIn(BaseModel):
    text: str
    images: Optional[List[ImageInput]] = Field(default=None, description="Lista de imágenes en base64")

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/infer")
def infer(payload: InferenceIn):
    try:
        # construye contenido para el modelo
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": payload.text}]
        total_bytes = 0

        # procesar lista de imágenes (si hay)
        if payload.images:
            for img in payload.images:
                if not img.image_b64.strip():
                    continue

                try:
                    img_bytes = base64.b64decode(img.image_b64, validate=True)
                except binascii.Error:
                    raise HTTPException(status_code=400, detail="Una de las imágenes no es base64 válida.")

                total_bytes += len(img_bytes)
                if total_bytes > MAX_REQ_BYTES:
                    raise HTTPException(status_code=413, detail="Demasiados datos (~>32 MiB en total).")

                if not img.mime:
                    import imghdr
                    fmt = imghdr.what(None, h=img_bytes)
                    img.mime = f"image/{fmt}" if fmt else "application/octet-stream"

                data_url = f"data:{img.mime};base64,{img.image_b64}"
                content.append({"type": "input_image", "image_url": data_url})

        # Llamar al modelo
        resp = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": content}],
        )

        # 🔹 Solo devolvemos el texto del modelo
        return resp.output_text

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
