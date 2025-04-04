FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# This is only working from a separate install right now
RUN pip install git+https://github.com/huggingface/diffusers

# Create model directory
RUN mkdir -p /app/model

# Download full model and cache
RUN python3 -c "\
from diffusers import WanImageToVideoPipeline; \
WanImageToVideoPipeline.from_pretrained('Wan-AI/Wan2.1-I2V-14B-480P-Diffusers')"

# Copy the API code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]