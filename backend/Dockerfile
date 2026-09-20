# Build context is the repo root (needs inference.py + root requirements.txt)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --timeout 180 --retries 10 -r backend/requirements.txt

COPY inference.py ./inference.py
COPY backend ./backend

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
