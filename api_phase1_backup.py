"""FastAPI backend for the Cervical AI Analysis research prototype.

Place this file beside the existing model checkpoints:
  phase2_resnet18_best.pth
  cervical_validator.pth
Run: uvicorn api:app --host 127.0.0.1 --port 8000
"""
from io import BytesIO
from pathlib import Path
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Type 1", "Type 2", "Type 3"]
EXPLANATIONS = {
    "Type 1": "The transformation zone is completely visible on the ectocervix.",
    "Type 2": "The transformation zone extends partly into the cervical canal, but it remains completely visible.",
    "Type 3": "The transformation zone extends into the cervical canal and is not completely visible.",
}
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI(
    title="Cervical AI Analysis API",
    description="Research prototype for cervical-image input screening, transformation-zone classification and Grad-CAM visualization.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def load_checkpoint(path: Path, output_classes: int) -> nn.Module:
    if not path.exists():
        raise RuntimeError(f"Model checkpoint not found: {path.name}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, output_classes)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


try:
    validator = load_checkpoint(VALIDATOR_PATH, 2)
    classifier = load_checkpoint(CLASSIFIER_PATH, 3)
    MODEL_ERROR = None
except Exception as exc:  # Report a helpful status; do not crash import/docs.
    validator = None
    classifier = None
    MODEL_ERROR = str(exc)


def make_gradcam(model: nn.Module, tensor: torch.Tensor, target_class: int) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = model.layer4[-1].conv2
    forward_handle = layer.register_forward_hook(lambda _m, _i, out: activations.append(out))
    backward_handle = layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))
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


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if MODEL_ERROR is None else "model_error",
        "device": str(DEVICE),
        "models_loaded": MODEL_ERROR is None,
        "detail": MODEL_ERROR,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    if MODEL_ERROR or validator is None or classifier is None:
        raise HTTPException(status_code=503, detail=f"Models are not ready: {MODEL_ERROR}")
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Upload a JPG or PNG image.")
    raw = await file.read()
    if not raw or len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is empty or exceeds the 20 MB limit.")
    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        validator_logits = validator(tensor)
        validator_probs = torch.softmax(validator_logits, dim=1)[0]
        is_cervical = int(torch.argmax(validator_probs).item()) == 1
    if not is_cervical:
        return {
            "valid_cervical_image": False,
            "message": "The input screening model did not classify this image as cervical. Try a suitable colposcopy image.",
            "screening_confidence": float(validator_probs.max().item()),
        }
    with torch.enable_grad():
        logits = classifier(tensor)
        probs = torch.softmax(logits.detach(), dim=1)[0]
        predicted = int(torch.argmax(probs).item())
        cam = make_gradcam(classifier, tensor, predicted)
    cam_small = cv2.resize(cam, (48, 48), interpolation=cv2.INTER_CUBIC)
    cam_small = np.clip(cam_small, 0.0, 1.0)
    name = CLASS_NAMES[predicted]
    next_step = (
        "Discuss the result with the clinician who performed the examination. They can decide whether further assessment is appropriate using the full clinical findings."
        if name == "Type 3" else
        "Discuss this result with your examining clinician alongside the full examination and any other test results."
    )
    return {
        "valid_cervical_image": True,
        "predicted_type": name,
        "predicted_index": predicted,
        "confidence": float(probs[predicted].item()),
        "probabilities": {CLASS_NAMES[i]: float(probs[i].item()) for i in range(3)},
        "interpretation": EXPLANATIONS[name],
        "suggested_next_step": next_step,
        "gradcam": cam_small.tolist(),
        "disclaimer": "Research prototype only. Transformation-zone type is an anatomical visibility category, not a cancer stage or a diagnosis. Grad-CAM shows model influence, not a confirmed lesion. A qualified clinician must interpret the complete clinical findings.",
    }
