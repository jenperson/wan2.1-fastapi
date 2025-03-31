# Use an official PyTorch base image with CUDA support (adjust tag for your GPU)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the WAN 2.1 model from Hugging Face
RUN python3 -c "from diffusers import WanPipeline; WanPipeline.from_pretrained('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', torch_dtype='auto')"

# Copy the API code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]