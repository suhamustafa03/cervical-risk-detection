# Data

This folder holds small, tracked files the app and scripts reference directly
(`facilities.json`, `phase2_split.csv`, `demo_sample.jpg`,
`annotated_references/`, `validator_negative/`). The two large raw training
sources below are **not committed to Git** (see `.gitignore`) because they
are large, publicly re-obtainable, and not needed to run or deploy the app
(`.vercelignore` excludes all of `data/` except `facilities.json` from the
deployed bundle regardless).

## `data/21442367/` — primary training/annotation dataset (not committed)

**INTELKAMAT – Cervical Colposcopy Image Dataset**
Rangelov, D., Prandzhev, G. D., & Miltchev, R.
Published on Zenodo, July 19, 2026 (v1). DOI: [10.5281/zenodo.21442367](https://doi.org/10.5281/zenodo.21442367)
License: **CC BY 4.0** (Creative Commons Attribution 4.0 International) — redistribution/reuse permitted with attribution to the creators.

934 anonymised RGB colposcopy JPEGs (1280×960) from 30 patients (`MUP_001`–`MUP_030`),
collected at the Medical University Hospital of Pleven, Bulgaria (2022–2024),
with transformation-zone type and clinical-finding annotations
(`patients.csv`, `images.csv`, `annotation_audit.csv`).

To use it locally: download the dataset from the DOI link above and extract
it into `data/21442367/` with the same internal structure
(`images/`, `annotated_references/`, `patients.csv`, `images.csv`,
`annotation_audit.csv`, `abbreviations.txt`). `make_split.py`,
`train_validator.py`, `phase2_model.py`, `confusion_matrix.py`, and the
Grad-CAM scripts (`gradcam.py`, `gradcam_multi.py`, `review2_gradcam.py`)
read from this path.

Cite the dataset per its Zenodo page if you publish results derived from it.

## `data/cifar-10-batches-py/` and `data/cifar-10-python.tar.gz` — validator negative examples (not committed)

The standard [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset
(Krizhevsky, 2009), used only as out-of-distribution/negative examples so the
image-validator model (`train_validator.py`) learns to reject non-cervix
images before they reach the classifier. `train_validator.py` downloads this
automatically via `torchvision.datasets.CIFAR10(..., download=True)` — no
manual setup needed; it is not re-downloaded if already cached locally.

## Why these aren't in Git

Both of the above are large (collectively ~670 MB) and already obtainable
from their original public sources, so committing them would bloat the
repository without adding anything a download step + citation doesn't
already provide. They are also irrelevant to the deployed app itself, which
only needs the trained model checkpoints (`cervical_validator.pth`,
`phase2_resnet18_best.pth`) and `data/facilities.json`.
