import os
import random
import torch

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models

# ============================================================
# PATHS
# ============================================================

CSV_PATH = "data/phase2_split.csv"
IMAGE_ROOT = "data/21442367/images"

MODEL_PATH = "cervical_validator.pth"

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("Using device:", device)

# ============================================================
# LOAD CERVICAL DATA
# ============================================================

import pandas as pd

df = pd.read_csv(CSV_PATH)

df = df[
    df["patient_tz_type"].isin(
        ["Type 1", "Type 2", "Type 3"]
    )
].copy()

# Use only training images as positive examples
cervical_df = df[
    df["split"] == "train"
].copy()

print(
    "Cervical training images:",
    len(cervical_df)
)

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
# CUSTOM DATASET
# ============================================================

class ValidatorDataset(Dataset):

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = Image.open(
            path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# CREATE POSITIVE SAMPLES
# ============================================================

positive_samples = []

for _, row in cervical_df.iterrows():

    path = os.path.join(
        IMAGE_ROOT,
        row["patient_id"],
        row["image_id"]
    )

    if os.path.exists(path):

        positive_samples.append(
            (path, 1)
        )


# ============================================================
# CREATE NON-CERVICAL SAMPLES
# USING CIFAR-10
# ============================================================

print("\nDownloading/loading CIFAR-10...")

from torchvision.datasets import CIFAR10

cifar_dataset = CIFAR10(
    root="data",
    train=True,
    download=True
)

# Randomly select the same number as cervical images
random.seed(42)

negative_indices = random.sample(
    range(len(cifar_dataset)),
    len(positive_samples)
)

negative_samples = []

for idx in negative_indices:

    image, _ = cifar_dataset[idx]

    # Save temporary image
    temp_path = os.path.join(
        "data",
        "validator_negative",
        f"negative_{idx}.png"
    )

    os.makedirs(
        os.path.dirname(temp_path),
        exist_ok=True
    )

    image.save(temp_path)

    negative_samples.append(
        (temp_path, 0)
    )


# ============================================================
# COMBINE DATA
# ============================================================

samples = (
    positive_samples +
    negative_samples
)

random.shuffle(samples)

print(
    "Total validator images:",
    len(samples)
)

print(
    "Cervical:",
    len(positive_samples)
)

print(
    "Non-cervical:",
    len(negative_samples)
)

# ============================================================
# DATASET / DATALOADER
# ============================================================

dataset = ValidatorDataset(
    samples,
    transform
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

# ============================================================
# RESNET18
# ============================================================

print("\nLoading ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)

# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

# ============================================================
# TRAIN
# ============================================================

EPOCHS = 3

for epoch in range(EPOCHS):

    model.train()

    correct = 0
    total = 0
    running_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    accuracy = (
        correct / total
    ) * 100

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {running_loss / len(loader):.4f} "
        f"- Accuracy: {accuracy:.2f}%"
    )


# ============================================================
# SAVE
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\n==============================")
print("VALIDATOR TRAINING COMPLETE")
print("==============================")

print(
    "Saved:",
    MODEL_PATH
)
