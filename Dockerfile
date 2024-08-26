FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
  build-essential \
  libatlas-base-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . /app

EXPOSE 8000
ENV PYTHONPATH=/app/src
CMD ["python", "src/main.py"]