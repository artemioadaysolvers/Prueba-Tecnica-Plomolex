FROM python:3.9-slim

WORKDIR /app

# Copiamos todo el repo
COPY . .

# Instalamos dependencias (requirements.txt en la raíz del repo)
RUN pip install --no-cache-dir -r requirements.txt

# Vamos a trabajar desde /app/gpt-proxy
WORKDIR /app/gpt-proxy

# (Opcional) Copiar index.html a /app/gpt-proxy/static/index.html si existe en la raíz del repo
RUN mkdir -p /app/gpt-proxy/static && \
    if [ -f /app/index.html ]; then cp /app/index.html /app/gpt-proxy/static/index.html; fi

ENV PORT=8080
EXPOSE 8080

# Import correcto: gpt-proxy/app/main.py => app.main:app
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
