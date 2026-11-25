import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

CSV_PATH = "data/phase2_split.csv"
IMAGE_ROOT = "data/21442367/images"
MODEL_PATH = "phase2_resnet18_best.pth"

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")
print("Using device:", device)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

label_map = {
    "Type 1": 0,
    "Type 2": 1,
    "Type 3": 2
}

df = df[df["patient_tz_type"].isin(label_map)].copy()
df["label"] = df["patient_tz_type"].map(label_map)

print("Total images:", len(df))
print("Classes:", list(label_map.keys()))

# ============================================================
# DATASET
# ============================================================

class CervicalDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = os.path.join(
            IMAGE_ROOT,
            row["patient_id"],
            row["image_id"]
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label"])

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# DATA SPLITS
# ============================================================

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_df = df[df["split"] == "test"]

train_dataset = CervicalDataset(
    train_df,
    train_transform
)

val_dataset = CervicalDataset(
    val_df,
    eval_transform
)

test_dataset = CervicalDataset(
    test_df,
    eval_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

# ============================================================
# RESNET18
# ============================================================

print("Loading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

# Freeze all pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Fine-tune final ResNet block
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final classifier
model.fc = nn.Linear(
    model.fc.in_features,
    3
)

model = model.to(device)

# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=0.0001
)

# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

# ============================================================
# TRAINING
# ============================================================

EPOCHS = 5

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    # ----------------------------
    # TRAIN
    # ----------------------------

    model.train()

    running_loss = 0.0

    train_predictions = []
    train_labels = []

    for images, labels in train_loader:

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

        train_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        train_labels.extend(
            labels.detach().cpu().numpy()
        )

    avg_train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = accuracy_score(
        train_labels,
        train_predictions
    )

    # ----------------------------
    # VALIDATION
    # ----------------------------

    model.eval()

    val_predictions = []
    val_labels = []

    running_val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_val_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_labels.extend(
                labels.cpu().numpy()
            )

    avg_val_loss = (
        running_val_loss / len(val_loader)
    )

    val_accuracy = accuracy_score(
        val_labels,
        val_predictions
    )

    # ----------------------------
    # STORE HISTORY
    # ----------------------------

    train_losses.append(
        avg_train_loss
    )

    val_losses.append(
        avg_val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Train Loss: {avg_train_loss:.4f} "
        f"- Train Accuracy: {train_accuracy * 100:.2f}% "
        f"- Val Loss: {avg_val_loss:.4f} "
        f"- Val Accuracy: {val_accuracy * 100:.2f}%"
    )

    # ----------------------------
    # SAVE BEST MODEL
    # ----------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Saved best model.")

# ============================================================
# ACCURACY GRAPH
# ============================================================

epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    [x * 100 for x in train_accuracies],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    [x * 100 for x in val_accuracies],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "Review 2 ResNet18 Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "phase2_accuracy.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Review 2 ResNet18 Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "phase2_loss.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

# ============================================================
# TEST EVALUATION
# ============================================================

test_predictions = []
test_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        test_predictions.extend(
            predictions.cpu().numpy()
        )

        test_labels.extend(
            labels.numpy()
        )

# ============================================================
# TEST ACCURACY
# ============================================================

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

print("\n==============================")
print("REVIEW 2 TEST RESULTS")
print("==============================")

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=[
            "Type 1",
            "Type 2",
            "Type 3"
        ],
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)

print("\nConfusion Matrix:")
print(cm)

# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Type 1",
        "Type 2",
        "Type 3"
    ]
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Review 2 Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "phase2_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# SAVED FILES
# ============================================================

print("\n==============================")
print("SAVED FILES")
print("==============================")

print(MODEL_PATH)
print("phase2_accuracy.png")
print("phase2_loss.png")
print("phase2_confusion_matrix.png")
