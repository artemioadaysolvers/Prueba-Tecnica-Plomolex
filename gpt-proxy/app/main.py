import os
import io
import base64
import binascii
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from pypdf import PdfReader

# -----------------------------
# Config
# -----------------------------
MODEL = os.getenv("MODEL", "gpt-4.1-mini")
MAX_REQ_BYTES = 32 * 1024 * 1024   # 32 MiB
MAX_PDF_TEXT_CHARS = 120_000       # recorte texto PDF

app = FastAPI(title="GPT Proxy", version="3.8")

# -----------------------------
# Paths (Dockerfile copia gpt-proxy/* a /app/*)
# /app/app/main.py
# /app/static/index.html
# -----------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))   # /app/app
BASE_DIR = os.path.dirname(APP_DIR)                    # /app
STATIC_DIR = os.path.join(BASE_DIR, "static")          # /app/static


# -----------------------------
# Models
# -----------------------------
class ImageInput(BaseModel):
    image_b64: str = Field(..., description="Imagen en base64 sin prefijo data:")
    mime: Optional[str] = Field(None, description="image/png, image/jpeg, etc.")


class InferenceIn(BaseModel):
    text: str
    images: Optional[List[ImageInput]] = None


# -----------------------------
# Helpers
# -----------------------------
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no está configurada")
    return OpenAI(api_key=api_key)


def read_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts: List[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        text = "\n\n".join(parts).strip()
        if not text:
            return ""
        if len(text) > MAX_PDF_TEXT_CHARS:
            text = text[:MAX_PDF_TEXT_CHARS] + "\n\n[...texto recortado por tamaño...]"
        return text
    except Exception:
        return ""


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="No se encontró static/index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "cwd": os.getcwd(),
        "base_dir": BASE_DIR,
        "static_dir": STATIC_DIR,
        "static_exists": os.path.exists(STATIC_DIR),
        "index_exists": os.path.exists(os.path.join(STATIC_DIR, "index.html")),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/infer")
def infer(payload: InferenceIn):
    client = get_openai_client()

    content: List[Dict[str, Any]] = [{"type": "input_text", "text": payload.text}]
    total_bytes = 0

    if payload.images:
        for img in payload.images:
            if not img.image_b64 or not img.image_b64.strip():
                continue

            try:
                img_bytes = base64.b64decode(img.image_b64, validate=True)
            except binascii.Error:
                raise HTTPException(status_code=400, detail="Imagen base64 inválida")

            total_bytes += len(img_bytes)
            if total_bytes > MAX_REQ_BYTES:
                raise HTTPException(status_code=413, detail="Payload demasiado grande (>32MiB)")

            mime = img.mime or "image/png"
            content.append({
                "type": "input_image",
                "image_url": f"data:{mime};base64,{img.image_b64}"
            })

    try:
        resp = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": content}],
        )
        return {"output": resp.output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.post("/summarize_pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Falta el archivo PDF")

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="El archivo debe ser PDF")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > MAX_REQ_BYTES:
        raise HTTPException(status_code=413, detail="PDF demasiado grande (>32MiB)")

    text = read_pdf_text(pdf_bytes)
    if not text:
        raise HTTPException(status_code=400, detail="No pude extraer texto del PDF (puede ser escaneado)")

    client = get_openai_client()

    prompt = (
        "Resume el siguiente documento en EXACTAMENTE 1 párrafo, en español, claro y directo. "
        "No uses viñetas ni listas.\n\n"
        f"{text}"
    )

    try:
        resp = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        )
        return {"summary": resp.output_text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarize error: {e}")
