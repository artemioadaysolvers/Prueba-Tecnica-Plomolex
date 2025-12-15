FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Cambia el working dir a la carpeta que contiene "app/"
WORKDIR /app/gpt-proxy

# (Opcional) Si tu index.html está en la raíz del repo, lo copiamos acá
RUN mkdir -p /app/gpt-proxy/app/static && \
    if [ -f /app/index.html ]; then cp /app/index.html /app/gpt-proxy/app/static/index.html; fi

ENV PORT=8080
EXPOSE 8080

# Ahora sí existe el módulo "app"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

