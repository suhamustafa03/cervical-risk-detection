"""Demo Setup & Patient Seeding Script for Cervical Risk Detection System.

Usage:
    python3 scripts/setup_demo.py
"""
import sys
from pathlib import Path
import requests

API_BASE = "http://127.0.0.1:8000"

def setup_demo():
    print("=== Setting up Live Demo Environment ===")
    
    # 1. Check Backend
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code != 200:
            print("[ERROR] Backend health check failed.")
            sys.exit(1)
        print("[OK] Backend service is running and models are ready.")
    except Exception as e:
        print(f"[ERROR] Could not connect to {API_BASE}: {e}")
        print("Please ensure uvicorn is running: uvicorn api:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    # 2. Seed Demo Patient
    demo_patient = {
        "first_name": "Ananya",
        "last_name": "Sharma",
        "age": 32,
        "gender": "Female",
        "case_ref": "REF-2026-DEMO-01",
        "notes": "Routine secondary cervical colposcopy screening; no prior excisional treatment.",
        "consent": True
    }
    
    try:
        p_res = requests.post(f"{API_BASE}/api/patients", json=demo_patient)
        if p_res.status_code != 201:
            print(f"[ERROR] Failed to seed patient: {p_res.text}")
            sys.exit(1)
        patient_data = p_res.json()
        patient_id = patient_data["id"]
        print(f"[OK] Demo patient created successfully:")
        print(f"     Name:       {patient_data['name']}")
        print(f"     Patient ID: {patient_id}")
        print(f"     Case Ref:   {demo_patient['case_ref']}")
    except Exception as e:
        print(f"[ERROR] Failed to create patient: {e}")
        sys.exit(1)

    # 3. Verify Demo Image Asset
    demo_img_path = Path("data/demo_sample.jpg")
    if demo_img_path.exists():
        print(f"[OK] Sample image asset confirmed at: {demo_img_path} ({demo_img_path.stat().st_size // 1024} KB)")
    else:
        print("[WARNING] data/demo_sample.jpg not found.")

    print("\n=== Demo Environment Primed & Ready ===")
    print("Open the web application in your browser: http://localhost:5173")
    print("Follow the presentation guide in: docs/demo_script.md\n")

if __name__ == "__main__":
    setup_demo()
