FROM python:3.9-slim

WORKDIR /app

# Install dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all datasets into the image for distributed use
COPY data/ /app/data/
# Create output and backup directories for in-container use
RUN mkdir -p /app/output /app/output/models /app/output/plots /app/output_backup /app/data_backup

# Copy application code
COPY . .

# Expose ports for Ray head node and API
EXPOSE 8000 8265 10001 6379

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command to run the API server
CMD ["python", "main.py", "--mode", "serve", "--host", "0.0.0.0", "--port", "8000"]
