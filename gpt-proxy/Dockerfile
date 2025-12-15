FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# (Opcional pero recomendado) que exista /static para el frontend
RUN mkdir -p /app/static && \
    if [ -f /app/index.html ]; then cp /app/index.html /app/static/index.html; fi

# Cloud Run usa la variable PORT
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
