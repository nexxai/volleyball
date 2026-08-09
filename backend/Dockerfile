FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.lock /app/
RUN pip install --no-cache-dir -r /app/requirements.lock

# Copy backend code
COPY backend/ /app/backend/
COPY ai_core/ /app/ai_core/

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/results /app/models /app/static

# Expose port
EXPOSE 8000

# Set working directory to backend
WORKDIR /app/backend

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
