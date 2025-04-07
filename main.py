from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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
STATUS_DIR = os.path.join(BASE_DIR, "job_status")
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(STATUS_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://localhost:5173"] for tighter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def run_video_job(job_id: str, request: GenerationRequest):
    try:
        image = load_image(request.image_url)

        max_area = 480 * 832
        aspect_ratio = image.height / image.width
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))

        frames = pipe(
            image=image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_frames=request.num_frames,
            guidance_scale=5.0,
        ).frames[0]

        video_filename = os.path.join(VIDEO_DIR, f"{job_id}.mp4")
        export_to_video(frames, video_filename, fps=16)

        # Mark job as complete
        with open(os.path.join(STATUS_DIR, f"{job_id}.done"), "w") as f:
            f.write("done")

    except Exception as e:
        with open(os.path.join(STATUS_DIR, f"{job_id}.error"), "w") as f:
            f.write(str(e))

@app.post("/generate-video")
def generate_video(request: GenerationRequest, background_tasks: BackgroundTasks):
    print(f"request created: {request.prompt}, {request.negative_prompt}, {request.num_frames}, {request.image_url}")
    job_id = str(uuid.uuid4())

    background_tasks.add_task(run_video_job, job_id, request)

    return {"job_id": job_id, "status_url": f"/status/{job_id}", "video_url": f"/videos/{job_id}.mp4"}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if os.path.exists(os.path.join(STATUS_DIR, f"{job_id}.done")):
        return {"status": "completed", "video_url": f"/videos/{job_id}.mp4"}
    elif os.path.exists(os.path.join(STATUS_DIR, f"{job_id}.error")):
        with open(os.path.join(STATUS_DIR, f"{job_id}.error"), "r") as f:
            return {"status": "error", "detail": f.read()}
    else:
        return {"status": "processing"}

@app.get("/")
def root():
    return {"message": "WAN 2.1 FastAPI service is running!"}