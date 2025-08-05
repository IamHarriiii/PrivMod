FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libgomp1 build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY server/requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy server code
COPY server/server.py .

# Run server
CMD ["python", "server.py"]