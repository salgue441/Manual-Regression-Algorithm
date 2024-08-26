FROM python:3.12.0-slim-buster
WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
  build-essential \
  libatlas-base-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000

ENV PYTHONPATH=/app/src
CMD ["python", "src/main.py"]