import os
import pandas as pd
import torch
import numpy as np
from torch import nn
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
CSV_PATH = "data/phase2_split.csv"
IMAGE_ROOT = "data/21442367/images"
MODEL_PATH = "phase2_resnet18_best.pth"

# -----------------------------
# Device
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# Load dataset split
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Use the first TEST image for the Grad-CAM demonstration
test_row = df[df["split"] == "test"].iloc[0]

image_path = os.path.join(
    IMAGE_ROOT,
    test_row["patient_id"],
    test_row["image_id"]
)

print("Demo image:", image_path)
print("True Type:", test_row["patient_tz_type"])

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load image
# -----------------------------
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)
input_tensor.requires_grad_(True)

# -----------------------------
# Load model
# -----------------------------
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

# -----------------------------
# Grad-CAM hooks
# -----------------------------
activations = []
gradients = []

target_layer = model.layer4[-1].conv2

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# -----------------------------
# Forward pass
# -----------------------------
output = model(input_tensor)

predicted_class = torch.argmax(output, dim=1).item()

class_names = [
    "Type 1",
    "Type 2",
    "Type 3"
]

print("Predicted Type:", class_names[predicted_class])

# -----------------------------
# Backward pass
# -----------------------------
model.zero_grad()

score = output[0, predicted_class]
score.backward()

# -----------------------------
# Generate Grad-CAM
# -----------------------------
activation = activations[0].detach()
gradient = gradients[0].detach()

weights_cam = gradient.mean(
    dim=(2, 3),
    keepdim=True
)

cam = (weights_cam * activation).sum(dim=1).squeeze()

cam = torch.relu(cam)

cam = cam.cpu().numpy()

# Normalize CAM
cam -= cam.min()

if cam.max() > 0:
    cam /= cam.max()

# Resize heatmap
cam_image = Image.fromarray(
    np.uint8(cam * 255)
).resize(image.size)

heatmap = np.array(cam_image)

# -----------------------------
# Create visualization
# -----------------------------
original = np.array(image)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title(
    f"Original Image\nPredicted: {class_names[predicted_class]}"
)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(original)
plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.45
)
plt.title("Grad-CAM Explanation")
plt.axis("off")

plt.tight_layout()

# Save result
output_file = "gradcam_result.png"
plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nGrad-CAM saved as:", output_file)
