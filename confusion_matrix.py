import pandas as pd
import torch
from torch import nn
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/phase2_split.csv")
df = df[df["split"] == "test"].copy()

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(
    torch.load(
        "phase2_resnet18_best.pth",
        map_location=device
    )
)

model.eval()

classes = ["Type 1", "Type 2", "Type 3"]

true_labels = []
pred_labels = []

for _, row in df.iterrows():

    path = os.path.join(
        "data/21442367/images",
        row["patient_id"],
        row["image_id"]
    )

    image = Image.open(path).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        prediction = torch.argmax(output, dim=1).item()

    true_labels.append(classes.index(row["patient_tz_type"]))
    pred_labels.append(prediction)

cm = confusion_matrix(true_labels, pred_labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot()

plt.title("Phase 2 ResNet18 Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "phase2_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("Saved: phase2_confusion_matrix.png")
