import torch
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from tensorflow.keras.models import load_model

model = load_model("bg_remover/artifacts/model")
app=FastAPI()
@app.post("/removebg")
def remove_baground(prompt: ):
    input_ids = text_to_token_ids(prompt.text, tokenizer).to(device)
    output_ids = model.generate(input_ids, max_new_tokens=prompt.max_new_tokens)
    output_text = token_ids_to_text(output_ids, tokenizer)
    return {"output": output_text}
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "device": str(device)
    }
    