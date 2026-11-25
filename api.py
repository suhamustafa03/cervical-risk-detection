"""FastAPI backend for the cervical colposcopy research prototype.

The ResNet18 architectures, state-dict weights, class ordering, and preprocessing
are preserved. Model construction is lazy so non-inference requests do not pay
the model-load cost in a serverless instance.
"""
from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import threading
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
VALIDATOR_PATH = BASE_DIR / "cervical_validator.pth"
CLASSIFIER_PATH = BASE_DIR / "phase2_resnet18_best.pth"
IS_VERCEL = os.getenv("VERCEL", "").lower() in {"1", "true"} or bool(os.getenv("VERCEL_ENV"))
DEVICE = torch.device("cpu" if IS_VERCEL else ("cuda" if torch.cuda.is_available() else "cpu"))
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
CLASS_NAMES = ["Type 1", "Type 2", "Type 3"]
EXPLANATIONS = {
    "Type 1": "The transformation zone is completely visible on the ectocervix.",
    "Type 2": "The transformation zone has an endocervical component but is fully visible.",
    "Type 3": "The transformation zone is not fully visible / extends into the canal.",
}
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI(
    title="Cervical AI Analysis API",
    description=(
        "Research prototype for cervical-image input screening, transformation-zone "
        "classification and Grad-CAM visualization. Not a diagnostic device."
    ),
    version="1.1.0",
)

default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]
env_origins = os.getenv("ALLOWED_ORIGINS", os.getenv("FRONTEND_ORIGIN", ""))
extra_origins = [origin.strip() for origin in env_origins.split(",") if origin.strip()]
origins = list(dict.fromkeys(default_origins + extra_origins))
cors_origin_regex = None if IS_VERCEL else os.getenv("CORS_ORIGIN_REGEX", r"^https://.*\.vercel\.app$")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_MODEL_LOCK = threading.Lock()
_MODELS: tuple[nn.Module, nn.Module] | None = None
_MODEL_ERROR: str | None = None


def load_checkpoint(path: Path, output_classes: int) -> nn.Module:
    if not path.is_file():
        raise RuntimeError(f"Model checkpoint not found: {path.name}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, output_classes)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def get_models() -> tuple[nn.Module, nn.Module]:
    global _MODELS, _MODEL_ERROR
    if _MODELS is not None:
        return _MODELS
    if _MODEL_ERROR is not None:
        raise HTTPException(status_code=503, detail=f"Models are not ready: {_MODEL_ERROR}")
    with _MODEL_LOCK:
        if _MODELS is not None:
            return _MODELS
        if _MODEL_ERROR is not None:
            raise HTTPException(status_code=503, detail=f"Models are not ready: {_MODEL_ERROR}")
        try:
            if IS_VERCEL:
                torch.set_num_threads(1)
            _MODELS = (
                load_checkpoint(VALIDATOR_PATH, 2),
                load_checkpoint(CLASSIFIER_PATH, 3),
            )
        except Exception as exc:
            _MODEL_ERROR = str(exc)
            raise HTTPException(status_code=503, detail=f"Models are not ready: {_MODEL_ERROR}") from exc
    return _MODELS


def make_gradcam(model: nn.Module, tensor: torch.Tensor, target_class: int) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = model.layer4[-1].conv2
    forward_handle = layer.register_forward_hook(lambda _module, _inputs, output: activations.append(output))
    backward_handle = layer.register_full_backward_hook(lambda _module, _grad_in, grad_out: gradients.append(grad_out[0]))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        logits[0, target_class].backward()
        if not activations or not gradients:
            raise RuntimeError("Could not generate Grad-CAM for this image.")
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[0]).sum(dim=1)).squeeze(0)
        cam = cam - cam.min()
        maximum = cam.max()
        if maximum > 0:
            cam = cam / maximum
        return cam.detach().cpu().numpy().astype(np.float32)
    finally:
        forward_handle.remove()
        backward_handle.remove()


def _decode_image(raw: bytes, content_type: str) -> Image.Image:
    if content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPG or PNG image.")
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is empty or exceeds the 20 MB limit.")
    try:
        with Image.open(BytesIO(raw)) as probe:
            image_format = probe.format
            if probe.width * probe.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions exceed the 20 megapixel limit.")
            probe.verify()
        expected = "PNG" if content_type == "image/png" else "JPEG"
        if image_format != expected:
            raise HTTPException(status_code=415, detail="Image content does not match its declared JPG or PNG format.")
        with Image.open(BytesIO(raw)) as decoded:
            decoded.load()
            return decoded.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable JPG or PNG image.") from exc


def analyze_bytes(raw: bytes, content_type: str) -> dict[str, Any]:
    image = _decode_image(raw, content_type)
    validator, classifier = get_models()
    img_arr = np.asarray(image, dtype=np.float32)
    if float(img_arr.std()) < 10.0:
        return {
            "valid_cervical_image": False,
            "message": "Invalid or unreadable image. Please provide a clear colposcopy photograph.",
            "screening_confidence": 0.0,
        }

    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        validator_logits = validator(tensor)
        validator_probs = torch.softmax(validator_logits, dim=1)[0]

    cervical_confidence = float(validator_probs[1].item())
    if cervical_confidence < 0.80:
        return {
            "valid_cervical_image": False,
            "message": "The input screening model did not classify this image as cervical. Try a suitable colposcopy image.",
            "screening_confidence": float(validator_probs.max().item()),
        }

    with torch.enable_grad():
        logits = classifier(tensor)
        probabilities = torch.softmax(logits.detach(), dim=1)[0]
        predicted = int(torch.argmax(probabilities).item())
        cam = make_gradcam(classifier, tensor, predicted)
    cam_small = cv2.resize(cam, (48, 48), interpolation=cv2.INTER_CUBIC)
    cam_small = np.clip(cam_small, 0.0, 1.0)
    name = CLASS_NAMES[predicted]
    next_step = (
        "Discuss the result with the clinician who performed the examination. They can decide whether further assessment is appropriate using the full clinical findings."
        if name == "Type 3"
        else "Discuss this result with your examining clinician alongside the full examination and any other test results."
    )
    return {
        "valid_cervical_image": True,
        "predicted_type": name,
        "predicted_index": predicted,
        "confidence": float(probabilities[predicted].item()),
        "probabilities": {CLASS_NAMES[index]: float(probabilities[index].item()) for index in range(3)},
        "interpretation": EXPLANATIONS[name],
        "suggested_next_step": next_step,
        "gradcam": cam_small.tolist(),
        "disclaimer": (
            "Research prototype only. Transformation-zone type is an anatomical visibility category, not a cancer stage or a diagnosis. "
            "Grad-CAM shows model influence, not a confirmed lesion. A qualified clinician must interpret the complete clinical findings."
        ),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    database_configured = bool(os.getenv("DATABASE_URL")) or not IS_VERCEL
    storage_configured = (
        STORAGE_BACKEND == "local" and not IS_VERCEL
    ) or (
        STORAGE_BACKEND == "blob" and bool(os.getenv("BLOB_READ_WRITE_TOKEN") or os.getenv("BLOB_STORE_ID"))
    )
    models_available = VALIDATOR_PATH.is_file() and CLASSIFIER_PATH.is_file()
    checks_ok = models_available and database_configured and storage_configured
    error = None
    if not models_available:
        error = "One or more model checkpoints are not included in the deployment."
    elif not database_configured:
        error = "Persistent PostgreSQL is not configured."
    elif not storage_configured:
        error = "Private Blob storage is not configured."
    return {
        "status": "ok" if checks_ok else "degraded",
        "device": str(DEVICE),
        "models_available": models_available,
        "models_loaded": _MODELS is not None,
        "storage_backend": STORAGE_BACKEND,
        "database_configured": database_configured,
        "detail": error or _MODEL_ERROR,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Upload a JPG or PNG image.")
    raw = await file.read(MAX_IMAGE_BYTES + 1)
    return analyze_bytes(raw, file.content_type)


import sys
from backend_addons import STORAGE_BACKEND, build_router

app.include_router(build_router(sys.modules[__name__]))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
