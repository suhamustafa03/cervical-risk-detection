import os
import pandas as pd
import torch
import numpy as np
from torch import nn
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt

CSV_PATH = "data/phase2_split.csv"
IMAGE_ROOT = "data/21442367/images"
MODEL_PATH = "phase2_resnet18_best.pth"

device = torch.device("cpu")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CSV_PATH)
test_df = df[df["split"] == "test"].copy()

classes = ["Type 1", "Type 2", "Type 3"]

# Select one image from a different patient for each type
selected = []

for class_name in classes:
    rows = test_df[test_df["patient_tz_type"] == class_name]

    if len(rows) == 0:
        raise ValueError(f"No test image found for {class_name}")

    # Use first image from first patient
    patient = rows.iloc[0]["patient_id"]
    row = rows[rows["patient_id"] == patient].iloc[0]
    selected.append(row)

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load model
# -----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

# Grad-CAM target layer
target_layer = model.layer4[-1].conv2

# -----------------------------
# Figure
# -----------------------------
fig, axes = plt.subplots(
    3,
    2,
    figsize=(12, 15)
)

for i, row in enumerate(selected):

    image_path = os.path.join(
        IMAGE_ROOT,
        row["patient_id"],
        row["image_id"]
    )

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    activations = []
    gradients = []

    def forward_hook(module, inp, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    output = model(input_tensor)

    predicted_class = torch.argmax(
        output,
        dim=1
    ).item()

    # Backward
    model.zero_grad()

    score = output[0, predicted_class]
    score.backward()

    # Remove hooks
    fh.remove()
    bh.remove()

    # -----------------------------
    # Create Grad-CAM
    # -----------------------------
    activation = activations[0].detach()
    gradient = gradients[0].detach()

    weights_cam = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights_cam * activation
    ).sum(dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam.cpu().numpy()

    cam -= cam.min()

    if cam.max() > 0:
        cam /= cam.max()

    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    ).resize(image.size)

    heatmap = np.array(cam_image)

    original = np.array(image)

    # -----------------------------
    # Original
    # -----------------------------
    axes[i, 0].imshow(original)

    axes[i, 0].set_title(
        f"{classes[predicted_class]}\n"
        f"True: {row['patient_tz_type']}"
    )

    axes[i, 0].axis("off")

    # -----------------------------
    # Grad-CAM
    # -----------------------------
    axes[i, 1].imshow(original)

    axes[i, 1].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )

    axes[i, 1].set_title(
        f"Grad-CAM\n"
        f"Predicted: {classes[predicted_class]}"
    )

    axes[i, 1].axis("off")

    print(
        f"{row['image_id']} | "
        f"True: {row['patient_tz_type']} | "
        f"Predicted: {classes[predicted_class]}"
    )

plt.suptitle(
    "Explainable Cervical Image Analysis using Grad-CAM",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "gradcam_three_types.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nSaved: gradcam_three_types.png")
