FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Expose port for API
EXPOSE 8000

# Start command with default argument
ENTRYPOINT ["python", "src/main.py"]
CMD ["--mode=serve", "--config=src/config/config.yaml"]
