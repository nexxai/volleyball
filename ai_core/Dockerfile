FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.lock /app/
RUN pip install --no-cache-dir -r /app/requirements.lock

# Copy AI core code
COPY ai_core/ /app/ai_core/

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/results /app/models

# Set working directory to ai_core
WORKDIR /app/ai_core

# Set environment variables
ENV PYTHONPATH=/app
ENV CELERY_BROKER_URL=redis://redis:6379/0
ENV CELERY_RESULT_BACKEND=redis://redis:6379/0

# Run Celery worker
CMD ["celery", "-A", "worker", "worker", "--loglevel=info"]
