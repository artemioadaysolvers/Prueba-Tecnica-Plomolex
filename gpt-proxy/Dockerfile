WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/gpt-proxy

ENV PORT=8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
