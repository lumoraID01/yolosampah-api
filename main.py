# --- TOP OF FILE (gantikan bagian import & CONFIG kamu) ---
# --- TOP OF FILE (gantikan bagian import & CONFIG kamu) ---
import os
import io
import base64
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import cv2
import torch
import numpy as np
from ultralytics import YOLO


# ============ CONFIG ============
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")  # <— FIXED
ALLOWED_ORIGINS = ["*"]  # ganti ke domain frontend di produksi
TOPK_DEFAULT = 3         # utk klasifikasi
# ===============================


# ---- App & CORS
app = FastAPI(title="YoloSampah API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model sekali
if not Path(MODEL_PATH).exists():
    raise RuntimeError(f"MODEL_PATH tidak ditemukan: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
TASK = model.task  # "classify" | "detect" | "segment"
NAMES = model.names

# ---- Schemas
class Box(BaseModel):
    x1: float; y1: float; x2: float; y2: float

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: Optional[Box] = None   # None untuk klasifikasi

class PredictResponse(BaseModel):
    task: str                   # classify/detect
    preds: List[Detection]
    annotated_image_b64: Optional[str] = None

# ---- Helpers
def _to_b64(image_bgr: np.ndarray) -> str:
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".jpg", img_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(500, "Gagal encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

# ---- Routes
@app.get("/health")
def health():
    return {
        "status": "ok",
        "task": TASK,
        "classes": NAMES,
        "model_path": str(MODEL_PATH)
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    topk: int = Query(TOPK_DEFAULT, ge=1, le=10),
    return_image: bool = Query(False),
):
    data = await image.read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    results = model.predict(pil, verbose=False)
    r = results[0]

    preds: List[Detection] = []

    if TASK == "classify":
        # top-k
        # ...
        probs = r.probs  # Ultralytics Probs obj
        # Ambil vector probabilitas (torch.Tensor atau np.ndarray)
        p = probs.data
        if isinstance(p, np.ndarray):
            # Numpy path
            k = min(topk, p.size)
            idxs = np.argsort(p)[::-1][:k]
            confs = p[idxs]
            idxs = idxs.tolist()
            confs = confs.tolist()
        elif isinstance(p, torch.Tensor):
            # Torch path
            p = p.float().cpu().flatten()
            k = min(topk, p.numel())
            vals, idxs = torch.topk(p, k)
            idxs = idxs.tolist()
            confs = vals.tolist()
        else:
            raise HTTPException(500, "Unknown probs.data type")

        for i, c in zip(idxs, confs):
            preds.append(Detection(
                class_id=int(i),
                class_name=NAMES[int(i)],
                confidence=float(c),
                box=None
            ))

        img_b64 = None
        if return_image:
            img_b64 = _to_b64(r.plot())


    elif TASK == "detect":
        # bbox list
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            preds.append(Detection(
                class_id=cls,
                class_name=NAMES[cls],
                confidence=conf,
                box=Box(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3])
            ))
        img_b64 = _to_b64(r.plot()) if return_image else None

    else:
        raise HTTPException(400, f"TASK '{TASK}' belum di-support API")

    return PredictResponse(task=TASK, preds=preds, annotated_image_b64=img_b64)
