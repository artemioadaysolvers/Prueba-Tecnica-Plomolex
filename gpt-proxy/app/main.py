import os
import base64
import binascii
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openai import OpenAI

from pypdf import PdfReader
import io

# -------------------------------------------------
# Configuración
# -------------------------------------------------
MODEL = os.getenv("MODEL", "gpt-4.1-mini")
MAX_REQ_BYTES = 32 * 1024 * 1024
MAX_PDF_TEXT_CHARS = 120_000

app = FastAPI(title="GPT Proxy", version="3.6-pdf")

# -------------------------------------------------
# Rutas reales
# gpt-proxy/app/main.py
# gpt-proxy/static/index.html
# -------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))     # gpt-proxy/app
BASE_DIR = os.path.dirname(APP_DIR)                      # gpt-proxy
STATIC_DIR = os.path.join(BASE_DIR, "static")            # gpt-proxy/static


# -------------------------------------------------
# Modelos
# -------------------------------------------------
class ImageInput(BaseModel):
    image_b64: str
    mime: Optional[str] = None


class InferenceIn(BaseModel):
    text: str
    images: Optional[List[ImageInput]] = None


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(500, "OPENAI_API_KEY no configurada")
    return OpenAI(api_key=api_key)


def read_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()

        if not text:
            return ""

        if len(text) > MAX_PDF_TEXT_CHARS:
            text = text[:MAX_PDF_TEXT_CHARS] + "\n\n[texto recortado]"
        return text
    except Exception:
        return ""


# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/")
def frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)

    raise HTTPException(404, "index.html no encontrado en /static")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "base_dir": BASE_DIR,
        "static_dir": STATIC_DIR,
        "static_exists": os.path.exists(STATIC_DIR),
    }


@app.post("/infer")
def infer(payload: InferenceIn):
    client = get_openai_client()

    content = [{"type": "input_text", "text": payload.text}]
    total_bytes = 0

    if payload.images:
        for img in payload.images:
            try:
                img_bytes = base64.b64decode(img.image_b64, validate=True)
            except binascii.Error:
                raise HTTPException(400, "Imagen base64 inválida")

            total_bytes += len(img_bytes)
            if total_bytes > MAX_REQ_BYTES:
                raise HTTPException(413, "Payload demasiado grande")

            mime = img.mime or "image/png"
            content.append({
                "type": "input_image",
                "image_url": f"data:{mime};base64,{img.image_b64}"
            })

    resp = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": content}],
    )

    return {"output": resp.output_text}


@app.post("/summarize_pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(400, "Debe ser PDF")

    pdf_bytes = await file.read()
    text = read_pdf_text(pdf_bytes)

    if not text:
        raise HTTPException(400, "No se pudo extraer texto del PDF")

    client = get_openai_client()

    prompt = (
        "Resume el siguiente documento en EXACTAMENTE un párrafo, en español:\n\n"
        f"{text}"
    )

    resp = client.responses.create(
        model=MODEL,
        input=[{
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}]
        }],
    )

    return {"summary": resp.output_text.strip()}
