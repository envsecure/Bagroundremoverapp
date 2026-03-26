import torch
from fastapi import FastAPI
from pathlib import Path
import yaml
from bg_remover.src.model.deeplab import deeplabv3_plus

ROOT_DIR = Path(__file__).resolve().parents[1]

with open(ROOT_DIR / "configs" / "model.yaml", "r") as f:
    params = yaml.safe_load(f)

# ── Load Model ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_plus((params["H"], params["W"], 3))
model.load_state_dict(torch.load(f"{params['model_save_path']}/my_model.pth", map_location=device))
model = model.to(device)
model.eval()

app = FastAPI()


@app.post("/removebg")
def remove_background():
    # TODO: implement inference endpoint
    pass


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "device": str(device),
    }