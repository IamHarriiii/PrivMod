FROM python:3.9-buster

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libgomp1 build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY client.py .
ENV PYTHONPATH=/app
CMD ["python", "client.py"]