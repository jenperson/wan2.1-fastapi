# Use an official PyTorch image with CUDA (ensures compatibility)
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

# Ensure dependencies are installed before model download
RUN python3 -c "import torch, diffusers, transformers"

# Pre-download the WAN 2.1 model from Hugging Face
RUN python3 -c "from diffusers import WanPipeline; \
    WanPipeline.from_pretrained('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', torch_dtype='auto')"

# Copy the API code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]