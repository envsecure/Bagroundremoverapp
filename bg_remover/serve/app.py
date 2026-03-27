import asyncio
import io
import uuid
import time
import os
import torch
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import yaml

from bg_remover.src.model.deeplab import deeplabv3_plus

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
with open(ROOT_DIR / "configs" / "model.yaml", "r") as f:
    params = yaml.safe_load(f)

H, W = params["H"], params["W"]

# ── GPU Setup ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Inference Queue ────────────────────────────────────────────────────────────
inference_queue: asyncio.Queue = asyncio.Queue()
queue_worker_task = None
queue_stats = {"processed": 0, "total_time": 0.0}


async def queue_worker(model):
    """
    Background worker — pulls jobs ONE AT A TIME from the queue.
    Guarantees sequential GPU usage, no OOM, no race conditions.
    """
    while True:
        job_id, input_tensor, future = await inference_queue.get()
        start = time.perf_counter()
        try:
            with torch.no_grad():
                output = model(input_tensor)
            future.set_result(output)
            elapsed = time.perf_counter() - start
            queue_stats["processed"] += 1
            queue_stats["total_time"] += elapsed
            print(f"  ✅ Job {job_id} done in {elapsed:.3f}s | Queue: {inference_queue.qsize()}")
        except Exception as e:
            future.set_exception(e)
        finally:
            inference_queue.task_done()


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global queue_worker_task

    model = deeplabv3_plus((H, W, 3))
    model_path = os.environ.get("MODEL_PATH", f"{params['model_save_path']}/my_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded from {model_path} on {device}")

    app.state.model = model

    queue_worker_task = asyncio.create_task(queue_worker(model))
    print("✅ Inference queue worker started")

    yield

    queue_worker_task.cancel()
    print("🛑 Queue worker stopped")


app = FastAPI(
    title="BG Remover API",
    description="AI Background Removal with queue-based GPU processing",
    lifespan=lifespan,
)

# ── CORS (for local dev) ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> torch.Tensor:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.resize(img, (W, H))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0)
    return tensor.to(device)


def postprocess(mask_tensor: torch.Tensor, original_bytes: bytes) -> bytes:
    arr = np.frombuffer(original_bytes, np.uint8)
    original = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    orig_h, orig_w = original.shape[:2]

    mask = mask_tensor.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (orig_w, orig_h))
    mask = (mask * 255).astype(np.uint8)

    b, g, r = cv2.split(original)
    rgba = cv2.merge([b, g, r, mask])

    _, png_bytes = cv2.imencode(".png", rgba)
    return png_bytes.tobytes()


# ── API Endpoints ──────────────────────────────────────────────────────────────
@app.post("/removebg")
async def remove_background(file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        input_tensor = preprocess(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    loop = asyncio.get_event_loop()
    future = loop.create_future()
    job_id = str(uuid.uuid4())[:8]

    await inference_queue.put((job_id, input_tensor, future))
    print(f"  📥 Job {job_id} queued (depth: {inference_queue.qsize()})")

    output_tensor = await future
    result_png = postprocess(output_tensor, image_bytes)

    return StreamingResponse(
        io.BytesIO(result_png),
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename=no_bg_{file.filename}"},
    )


@app.get("/queue")
async def queue_status():
    avg_time = (
        queue_stats["total_time"] / queue_stats["processed"]
        if queue_stats["processed"] > 0
        else 0
    )
    return {
        "pending_jobs": inference_queue.qsize(),
        "total_processed": queue_stats["processed"],
        "avg_inference_time_sec": round(avg_time, 3),
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "device": str(device),
        "queue_depth": inference_queue.qsize(),
    }


# ── Serve Frontend (production: static files from build) ───────────────────────
FRONTEND_DIR = ROOT_DIR.parent / "frontend" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Catch-all: serve index.html for any non-API route (SPA routing)."""
        file_path = FRONTEND_DIR / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))