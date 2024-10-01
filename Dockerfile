FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
COPY main.py /app/
COPY scrapper.py /app/
COPY main.py /app/
COPY api_keys /app/api_keys
COPY models /app/models

RUN pip install --no-cache-dir -r requirements.txt



CMD ["python", "main.py"]
