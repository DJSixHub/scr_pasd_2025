FROM python:3.9-slim

WORKDIR /app

# Install dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create output directories
RUN mkdir -p output/models output/plots

# Copy application code
COPY . .

# Expose ports for Ray head node and API
EXPOSE 8000 8265 10001 6379

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command to run the API server
CMD ["python", "main.py", "--mode", "serve", "--host", "0.0.0.0", "--port", "8000"]
