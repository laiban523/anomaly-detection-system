FROM python:3.11-slim

WORKDIR /app

# Ensure all services (including streamlit) resolve modules from /app
ENV PYTHONPATH=/app

# Install system deps for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ dir will be volume-mounted; create it so it exists in image too
RUN mkdir -p /app/data
