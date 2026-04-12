# Audio Service - Final Version
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Configure pip
ENV PIP_DEFAULT_TIMEOUT=600
ENV PIP_RETRY=10
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

# Copy requirements
COPY requirements-final.txt .
RUN pip install --timeout 600 --retries 10 --no-cache-dir -r requirements-final.txt

# Copy service code
COPY services/audio/main.py .

# Create temp directory
RUN mkdir -p temp_audio

ENV PORT=5004

EXPOSE 5004

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5004"]
