# GPT Proxy (Cloud Run)

Servicio HTTP que recibe texto y devuelve la respuesta de OpenAI con un modelo fijado por env var `MODEL`.
La API key se lee desde `OPENAI_API_KEY` (inyectada por Secret Manager en Cloud Run).

## Endpoints
- `GET /health` -> `{ status, model }`
- `POST /infer` -> body: `{ "text": "..." }` -> `{ "model": "...", "output": "..." }`

## Variables
- `MODEL` (opcional): por defecto `gpt-4o-mini`. Puedes usar `gpt-3.5-turbo` si prefieres.
- `OPENAI_API_KEY`: **no** debe ir en el repo; se configura con Secret Manager.

## Run local
