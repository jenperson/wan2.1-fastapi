from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import WanPipeline
from diffusers.utils import export_to_video
import torch
import uuid
import os
from PIL import Image
from fastapi.staticfiles import StaticFiles
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "generated_images")
VIDEO_DIR = os.path.join(BASE_DIR, "generated_videos")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

app = FastAPI()

app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# Load the model once at startup
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
print("Loading model...")
pipe = WanPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # Enable CPU offloading if limited GPU memory
print("Model loaded!")

# Define a request schema
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_frames: int = 33  # Only used for video

@app.post("/generate-video")
def generate_video(request: GenerationRequest):
    """Generate a video from a text prompt."""
    try:
        frames = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_frames=request.num_frames
        ).frames[0]

        video_filename = f"{VIDEO_DIR}/{uuid.uuid4()}.mp4"
        export_to_video(frames, video_filename, fps=16)

        return {"video_path": video_filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
def generate_image(request: GenerationRequest):
    """Generate a single image from a text prompt."""
    try:
        frames = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_frames=1  # Only generate 1 frame
        ).frames[0]

        image_filename = f"{IMAGE_DIR}/{uuid.uuid4()}.png"
        image = Image.fromarray(frames[0])  # Convert frame to PIL Image
        image.save(image_filename)

        return {"image_path": image_filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "WAN 2.1 FastAPI service is running!"}