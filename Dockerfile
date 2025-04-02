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

# Download model in chunks to leverage Docker layer caching
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['model_index.json'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['image_encoder/config.json'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['image_encoder/model.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['image_processor/preprocessor_config.json'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['scheduler/scheduler_config.json'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['text_encoder/config.json'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['text_encoder/model-00001-of-00005.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['text_encoder/model-00002-of-00005.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['text_encoder/model-00003-of-00005.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['text_encoder/model-00004-of-00005.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['text_encoder/model-00005-of-00005.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['text_encoder/model.safetensors.index.json'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['tokenizer/special_tokens_map.json'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['tokenizer/spiece.model'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['tokenizer/tokenizer.json'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['tokenizer/tokenizer_config.json'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/config.json'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00001-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00002-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00003-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00004-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00005-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00006-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00007-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00008-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00009-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00010-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00011-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00012-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00013-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model-00014-of-00014.safetensors'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['transformer/diffusion_pytorch_model.safetensors.index.json'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['vae/config.json'])"
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='/app/model', allow_patterns=['vae/diffusion_pytorch_model.safetensors'])"


# Pre-download the WAN 2.1 model from Hugging Face

# RUN python3 -c "from diffusers import WanPipeline; \
#     WanPipeline.from_pretrained('Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', torch_dtype='auto')"

# Copy the API code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]