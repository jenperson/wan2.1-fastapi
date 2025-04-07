from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
import uuid
import os
from PIL import Image
from fastapi.staticfiles import StaticFiles
import os

model_path = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

# Add directory to store generated videos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "generated_videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

app = FastAPI()

app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")

# Load the model from the container
image_encoder = CLIPVisionModel.from_pretrained(
    model_path, subfolder="image_encoder", torch_dtype=torch.float32
)
vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(
    model_path,
    vae=vae,
    image_encoder=image_encoder,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
print("Model loaded!")

# Define the request schema
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_frames: int = 33
    image_url: str = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"

@app.post("/generate-video")
def generate_video(request: GenerationRequest):
    print(f"request created: {request.prompt}, {request.negative_prompt}, {request.num_frames}, {request.image_url}")
    """Generate a video from a text prompt."""
    image = load_image(
        request.image_url
    )

    # See Hugging Face documentation for details: https://huggingface.co/docs/diffusers/main/en/api/pipelines/wan#image-to-video-generation
    max_area = 480 * 832
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))

    try:
        frames = pipe(
            image=image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_frames=request.num_frames,
            guidance_scale=5.0,
        ).frames[0]
        print("frames successfully generated")

        video_filename = f"{VIDEO_DIR}/{uuid.uuid4()}.mp4"
        export_to_video(frames, video_filename, fps=16)

        return {"video_path": video_filename}

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "WAN 2.1 FastAPI service is running!"}