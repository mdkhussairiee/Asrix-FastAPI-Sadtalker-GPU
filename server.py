import os
import subprocess
import tempfile
import shutil
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

# =========================================================
#  Configuration
# =========================================================

app = FastAPI(title="SadTalker GPU API", version="1.0")

# Security: token from environment (.env or docker-compose)
API_TOKEN = os.getenv("TALKING_HEAD_SERVICE_TOKEN", "my-secret-token")

# Directories
DATA_DIR = "/app/data"
RESULT_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Allow external access (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
#  Health check
# =========================================================

@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# =========================================================
#  Talking Head API
# =========================================================

@app.post("/v1/talking-head")
async def generate_talking_head(
    jobId: str = Form(...),
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    authorization: str = Header(None)
):
    """Main API: Generate talking-head video from image and audio."""

    # --- Auth check ---
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid authorization token")

    # --- Prepare temp directory ---
    job_dir = tempfile.mkdtemp(prefix=f"{jobId}_", dir=RESULT_DIR)
    img_path = os.path.join(job_dir, image.filename)
    audio_path = os.path.join(job_dir, audio.filename)
    await save_file(image, img_path)
    await save_file(audio, audio_path)

    # --- Prepare output path ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(job_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # --- Run SadTalker inference ---
    cmd = [
        "python3", "inference.py",
        "--driven_audio", audio_path,
        "--source_image", img_path,
        "--result_dir", output_dir,
        "--enhancer", "gfpgan",
        "--still",
        "--preprocess", "full"
    ]

    try:
        print(f"🚀 Running SadTalker job {jobId}")
        subprocess.run(cmd, check=True, cwd="/app")

        # Find the final generated video
        output_video = find_latest_mp4(output_dir)
        if not output_video:
            raise HTTPException(status_code=500, detail="Output video not found")

        print(f"✅ Generated video: {output_video}")
        return FileResponse(
            output_video,
            media_type="video/mp4",
            filename=f"{jobId}_output.mp4"
        )

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during inference: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")

    finally:
        # Optional cleanup (keep for debugging)
        cleanup_temp(job_dir, keep=True)


# =========================================================
#  Utility functions
# =========================================================

async def save_file(upload_file: UploadFile, destination: str):
    """Save an uploaded file asynchronously."""
    with open(destination, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)

def find_latest_mp4(directory: str) -> str:
    """Return the most recent .mp4 file from directory."""
    mp4_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(directory)
        for f in files if f.endswith(".mp4")
    ]
    return max(mp4_files, key=os.path.getmtime) if mp4_files else None

def cleanup_temp(directory: str, keep=False):
    """Delete or retain temporary job data."""
    if not keep and os.path.exists(directory):
        shutil.rmtree(directory, ignore_errors=True)


# =========================================================
#  Swagger UI / OpenAPI customization (auto Bearer token)
# =========================================================

def custom_openapi():
    """Inject BearerAuth into Swagger UI docs."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="SadTalker GPU API",
        version="1.0",
        description="Generate talking head videos using SadTalker (GPU-accelerated)",
        routes=app.routes,
    )

    # Add Bearer auth header globally
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "Token",
            "description": "Enter your API key as: **Bearer YOUR_TOKEN**"
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
