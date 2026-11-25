"""Synthetic-only smoke tests for the Vercel-ready FastAPI application.

The suite creates its own temporary SQLite database and media directory. It
never reads the project's existing database, storage, dataset, or test report.
"""
from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

_TEMP_DIR: tempfile.TemporaryDirectory[str] | None = None
_CLIENT: TestClient | None = None
_API = None
_OLD_ENV: dict[str, str | None] = {}


def setUpModule() -> None:
    global _TEMP_DIR, _CLIENT, _API
    for key in (
        "DATABASE_URL",
        "CERVICAL_STORAGE_DIR",
        "CERVICAL_STORAGE_BACKEND",
        "UPLOAD_SIGNING_SECRET",
        "VERCEL",
        "VERCEL_ENV",
    ):
        _OLD_ENV[key] = os.environ.get(key)
    os.environ.pop("VERCEL", None)
    os.environ.pop("VERCEL_ENV", None)
    _TEMP_DIR = tempfile.TemporaryDirectory(prefix="cervical-vercel-tests-")
    root = Path(_TEMP_DIR.name)
    os.environ["DATABASE_URL"] = f"sqlite:///{root / 'synthetic.sqlite3'}"
    os.environ["CERVICAL_STORAGE_DIR"] = str(root / "storage")
    os.environ["CERVICAL_STORAGE_BACKEND"] = "local"
    os.environ["UPLOAD_SIGNING_SECRET"] = "synthetic-only-test-secret-" + ("x" * 40)

    import api

    _API = api
    _CLIENT = TestClient(api.app)
    _CLIENT.__enter__()


def tearDownModule() -> None:
    if _CLIENT is not None:
        _CLIENT.__exit__(None, None, None)
    if _TEMP_DIR is not None:
        _TEMP_DIR.cleanup()
    for key, old_value in _OLD_ENV.items():
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


def png_bytes(array: np.ndarray | None = None) -> bytes:
    image = Image.new("RGB", (128, 128), (120, 90, 80)) if array is None else Image.fromarray(array)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def new_patient(client: TestClient, suffix: str) -> tuple[dict, dict[str, str]]:
    response = client.post(
        "/api/patients",
        json={
            "first_name": "Synthetic",
            "last_name": suffix,
            "age": 34,
            "gender": "Female",
            "case_ref": f"SYNTHETIC-{suffix}",
            "notes": "Generated test record only",
            "consent": True,
        },
    )
    if response.status_code != 201:
        raise AssertionError(response.text)
    patient = response.json()
    return patient, {"X-Patient-Access-Token": patient["access_token"]}


class VercelDeploymentSmokeTests(unittest.TestCase):
    @property
    def client(self) -> TestClient:
        assert _CLIENT is not None
        return _CLIENT

    def test_health_reports_bundled_models_and_local_test_services(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["models_available"])
        self.assertEqual(response.json()["storage_backend"], "local")

    def test_upload_rejects_declared_mime_that_disagrees_with_image(self) -> None:
        response = self.client.post(
            "/analyze",
            files={"file": ("synthetic.png", png_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 415)

    def test_real_model_checkpoints_load_and_blank_input_is_rejected(self) -> None:
        response = self.client.post(
            "/analyze",
            files={"file": ("synthetic-flat.png", png_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["valid_cervical_image"])
        self.assertTrue(self.client.get("/health").json()["models_loaded"])

    def test_real_classifier_and_gradcam_hooks_accept_synthetic_tensor(self) -> None:
        assert _API is not None
        import torch

        _, classifier = _API.get_models()
        pixels = np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        tensor = _API.PREPROCESS(Image.fromarray(pixels)).unsqueeze(0)
        with torch.no_grad():
            target = int(classifier(tensor).argmax(dim=1).item())
        cam = _API.make_gradcam(classifier, tensor, target)
        self.assertEqual(cam.shape, (7, 7))
        self.assertTrue(np.isfinite(cam).all())

    def test_patient_scan_history_and_capability_token_isolation(self) -> None:
        patient_a, token_a = new_patient(self.client, "A")
        patient_b, token_b = new_patient(self.client, "B")
        image = png_bytes()

        scan_a_response = self.client.post(
            f"/api/patients/{patient_a['id']}/scans",
            headers=token_a,
            files={"file": ("synthetic-a.png", image, "image/png")},
        )
        scan_b_response = self.client.post(
            f"/api/patients/{patient_b['id']}/scans",
            headers=token_b,
            files={"file": ("synthetic-b.png", image, "image/png")},
        )
        self.assertEqual(scan_a_response.status_code, 201, scan_a_response.text)
        self.assertEqual(scan_b_response.status_code, 201, scan_b_response.text)
        scan_a_id = scan_a_response.json()["scan"]["id"]
        scan_b_id = scan_b_response.json()["scan"]["id"]

        history_a = self.client.get(f"/api/patients/{patient_a['id']}/scans", headers=token_a)
        history_b = self.client.get(f"/api/patients/{patient_b['id']}/scans", headers=token_b)
        ids_a = {scan["id"] for scan in history_a.json()["scans"]}
        ids_b = {scan["id"] for scan in history_b.json()["scans"]}
        self.assertIn(scan_a_id, ids_a)
        self.assertNotIn(scan_b_id, ids_a)
        self.assertIn(scan_b_id, ids_b)
        self.assertNotIn(scan_a_id, ids_b)
        self.assertEqual(self.client.get(f"/api/scans/{scan_a_id}").status_code, 401)
        self.assertEqual(self.client.get(f"/api/scans/{scan_a_id}", headers=token_b).status_code, 403)

    def test_synthetic_pdf_report_persists_and_requires_patient_token(self) -> None:
        patient, token = new_patient(self.client, "REPORT")
        scan_response = self.client.post(
            f"/api/patients/{patient['id']}/scans",
            headers=token,
            files={"file": ("synthetic.png", png_bytes(), "image/png")},
        )
        self.assertEqual(scan_response.status_code, 201, scan_response.text)
        scan_id = scan_response.json()["scan"]["id"]
        pdf = b"%PDF-1.4\n% synthetic report test\n%%EOF\n"

        saved = self.client.post(
            f"/api/scans/{scan_id}/report",
            headers={**token, "Content-Type": "application/pdf"},
            content=pdf,
        )
        self.assertEqual(saved.status_code, 201, saved.text)
        retrieved = self.client.get(f"/api/scans/{scan_id}/report", headers=token)
        self.assertEqual(retrieved.status_code, 200)
        self.assertEqual(retrieved.content, pdf)
        self.assertEqual(self.client.get(f"/api/scans/{scan_id}/report").status_code, 401)

    def test_facility_filter_uses_bundled_public_directory(self) -> None:
        response = self.client.get("/api/facilities", params={"city": "Delhi"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(any("AIIMS" in row["name"] for row in response.json()["facilities"]))


if __name__ == "__main__":
    unittest.main()
