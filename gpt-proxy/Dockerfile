FROM python:3.9-slim

# Raíz del contenedor
WORKDIR /app

# Copiamos todo el repo
COPY . .

# Dependencias
RUN pip install --no-cache-dir -r requirements.txt

# MUY IMPORTANTE:
# Nos movemos a la carpeta que contiene "app/" y "static/"
WORKDIR /app/gpt-proxy

ENV PORT=8080
EXPOSE 8080

# Import correcto:
# gpt-proxy/app/main.py  →  app.main:app
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
