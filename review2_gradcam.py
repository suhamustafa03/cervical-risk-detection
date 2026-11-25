import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from PIL import Image
from torchvision import transforms, models


# ============================================================
# PATHS
# ============================================================

CSV_PATH = "data/phase2_split.csv"
IMAGE_ROOT = "data/21442367/images"
MODEL_PATH = "phase2_resnet18_best.pth"

OUTPUT_FILE = "review2_gradcam_results.png"


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "Type 1",
    "Type 2",
    "Type 3"
]


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

label_map = {
    "Type 1": 0,
    "Type 2": 1,
    "Type 3": 2
}

df = df[
    (df["split"] == "test") &
    (df["patient_tz_type"].isin(label_map))
].copy()

df["label"] = df["patient_tz_type"].map(label_map)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Review 2 ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(
    model.fc.in_features,
    3
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# ============================================================
# GRAD-CAM STORAGE
# ============================================================

activations = []
gradients = []

target_layer = model.layer4[-1].conv2


def forward_hook(module, input, output):
    activations.append(output)


def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])


target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# PREDICT FUNCTION
# ============================================================

def get_prediction(row):

    image_path = os.path.join(
        IMAGE_ROOT,
        row["patient_id"],
        row["image_id"]
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    output = model(input_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_class = torch.argmax(
        output,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    return (
        image,
        input_tensor,
        predicted_class,
        confidence,
        image_path
    )


# ============================================================
# FIND EXAMPLES
# ============================================================

correct_examples = {}

misclassified_example = None

print("\nSearching for examples...")


for _, row in df.iterrows():

    image, input_tensor, prediction, confidence, path = \
        get_prediction(row)

    actual = int(row["label"])

    if prediction == actual:

        if actual not in correct_examples:

            correct_examples[actual] = row

    else:

        if misclassified_example is None:

            misclassified_example = row

    if (
        len(correct_examples) == 3
        and misclassified_example is not None
    ):
        break


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(row):

    activations.clear()
    gradients.clear()

    image, input_tensor, predicted, confidence, path = \
        get_prediction(row)

    actual = int(row["label"])

    model.zero_grad()

    output = model(input_tensor)

    score = output[
        0,
        predicted
    ]

    score.backward()

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
    ).resize(
        image.size
    )

    heatmap = np.array(
        cam_image
    )

    return (
        image,
        heatmap,
        actual,
        predicted,
        confidence,
        path
    )


# ============================================================
# CREATE EXAMPLE LIST
# ============================================================

examples = []

for class_id in [0, 1, 2]:

    if class_id in correct_examples:

        examples.append(
            (
                f"{class_names[class_id]} - Correct",
                correct_examples[class_id]
            )
        )


if misclassified_example is not None:

    examples.append(
        (
            "Misclassified Example",
            misclassified_example
        )
    )


# ============================================================
# CREATE VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    len(examples),
    2,
    figsize=(12, 5 * len(examples))
)


if len(examples) == 1:
    axes = np.array([axes])


for i, (title, row) in enumerate(examples):

    image, heatmap, actual, predicted, confidence, path = \
        generate_gradcam(row)

    # Original image
    axes[i, 0].imshow(image)

    axes[i, 0].set_title(
        f"{title}\n"
        f"Actual: {class_names[actual]}\n"
        f"Predicted: {class_names[predicted]}"
    )

    axes[i, 0].axis("off")

    # Grad-CAM
    axes[i, 1].imshow(image)

    axes[i, 1].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45
    )

    axes[i, 1].set_title(
        f"Grad-CAM\n"
        f"Confidence: {confidence * 100:.2f}%"
    )

    axes[i, 1].axis("off")


plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\n==============================")
print("REVIEW 2 GRAD-CAM COMPLETE")
print("==============================")

print(
    "Saved:",
    OUTPUT_FILE
)
