FROM python:3.10-slim

# Dépendances système pour ML
RUN apt-get update && apt-get install -y \
    curl gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt.txt.txt

COPY . .
