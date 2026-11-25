"""Apply the project's additive patient/scan schema migration.

Set DATABASE_URL to the intended local or hosted database before running. This
script never imports or copies an existing database file into a deployment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if not os.getenv("DATABASE_URL"):
    raise SystemExit("Set DATABASE_URL explicitly before running this migration.")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend_addons import migrate_schema  # noqa: E402

migrate_schema()
print("Database schema is current; existing patient and scan rows were preserved.")
