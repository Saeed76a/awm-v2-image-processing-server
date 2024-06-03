FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install package
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python package
RUN pip install --no-cache-dir -r requirements.txt

# Go work directory
COPY . .

# Install package
RUN pip install faiss-cpu torch torchvision

# Go work directory
WORKDIR /app/image_recog

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9003"]