"""Automated test script for Phase 3: Patient Records & Scan History Isolation."""
import io
import requests
from PIL import Image

API_BASE = "http://127.0.0.1:8000"

def create_sample_colposcopy_like_image():
    """Create a temporary test image with sufficient variance to pass basic quality checks."""
    img = Image.new("RGB", (224, 224), color=(180, 80, 80))
    # Draw some varied pixels so std dev > 10
    pixels = img.load()
    for x in range(50, 150):
        for y in range(50, 150):
            pixels[x, y] = (220, 140, 140)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()

def run_tests():
    print("=== Phase 3: Patient & Scan Flow Verification ===")
    
    # 1. Age validation: ge=1, le=120
    print("\n1. Testing Age Validation...")
    res_age_low = requests.post(f"{API_BASE}/api/patients", json={
        "first_name": "Test", "last_name": "LowAge", "age": 0, "gender": "Female", "consent": True
    })
    assert res_age_low.status_code == 422, f"Expected 422 for age=0, got {res_age_low.status_code}"
    print("   [PASS] age=0 rejected (422)")

    res_age_high = requests.post(f"{API_BASE}/api/patients", json={
        "first_name": "Test", "last_name": "HighAge", "age": 121, "gender": "Female", "consent": True
    })
    assert res_age_high.status_code == 422, f"Expected 422 for age=121, got {res_age_high.status_code}"
    print("   [PASS] age=121 rejected (422)")

    # 2. Consent validation
    print("\n2. Testing Consent Validation...")
    res_no_consent = requests.post(f"{API_BASE}/api/patients", json={
        "first_name": "Test", "last_name": "NoConsent", "age": 30, "gender": "Female", "consent": False
    })
    assert res_no_consent.status_code == 400, f"Expected 400 for consent=False, got {res_no_consent.status_code}"
    print("   [PASS] consent=False rejected (400)")

    # 3. Create Patient A
    print("\n3. Creating Patient A and Patient B...")
    res_a = requests.post(f"{API_BASE}/api/patients", json={
        "first_name": "Alice", "last_name": "Smith", "age": 34, "gender": "Female", "case_ref": "REF-A", "consent": True
    })
    assert res_a.status_code == 201, f"Failed to create Patient A: {res_a.text}"
    patient_a = res_a.json()
    id_a = patient_a["id"]
    print(f"   [PASS] Created Patient A: {id_a} ({patient_a['name']})")

    # 4. Create Patient B
    res_b = requests.post(f"{API_BASE}/api/patients", json={
        "first_name": "Bobbie", "last_name": "Jones", "age": 42, "gender": "Female", "case_ref": "REF-B", "consent": True
    })
    assert res_b.status_code == 201, f"Failed to create Patient B: {res_b.text}"
    patient_b = res_b.json()
    id_b = patient_b["id"]
    print(f"   [PASS] Created Patient B: {id_b} ({patient_b['name']})")

    # 5. Upload Scan for Patient A
    print("\n4. Testing Scan Upload and Patient Isolation...")
    img_bytes = create_sample_colposcopy_like_image()
    scan_a_res = requests.post(
        f"{API_BASE}/api/patients/{id_a}/scans",
        files={"file": ("scan_a.jpg", img_bytes, "image/jpeg")}
    )
    assert scan_a_res.status_code == 201, f"Failed to upload scan for Patient A: {scan_a_res.text}"
    scan_a_id = scan_a_res.json()["scan"]["id"]
    print(f"   [PASS] Uploaded Scan for Patient A: Scan ID {scan_a_id}")

    # 6. Upload Scan for Patient B
    scan_b_res = requests.post(
        f"{API_BASE}/api/patients/{id_b}/scans",
        files={"file": ("scan_b.jpg", img_bytes, "image/jpeg")}
    )
    assert scan_b_res.status_code == 201, f"Failed to upload scan for Patient B: {scan_b_res.text}"
    scan_b_id = scan_b_res.json()["scan"]["id"]
    print(f"   [PASS] Uploaded Scan for Patient B: Scan ID {scan_b_id}")

    # 7. Verify Isolation: Patient A must only have scan_a
    history_a = requests.get(f"{API_BASE}/api/patients/{id_a}/scans").json()
    scans_a_ids = [s["id"] for s in history_a["scans"]]
    assert scan_a_id in scans_a_ids, "Patient A missing scan_a"
    assert scan_b_id not in scans_a_ids, "LEAK! Patient A history contains scan_b"
    print("   [PASS] Patient A scan history strictly contains only Patient A scans")

    # 8. Verify Isolation: Patient B must only have scan_b
    history_b = requests.get(f"{API_BASE}/api/patients/{id_b}/scans").json()
    scans_b_ids = [s["id"] for s in history_b["scans"]]
    assert scan_b_id in scans_b_ids, "Patient B missing scan_b"
    assert scan_a_id not in scans_b_ids, "LEAK! Patient B history contains scan_a"
    print("   [PASS] Patient B scan history strictly contains only Patient B scans")

    # 9. Test Single Scan Retrieval by Scan ID
    print("\n5. Testing Single Scan Retrieval...")
    get_scan_res = requests.get(f"{API_BASE}/api/scans/{scan_a_id}")
    assert get_scan_res.status_code == 200, f"Failed to retrieve scan {scan_a_id}: {get_scan_res.text}"
    retrieved = get_scan_res.json()
    assert retrieved["scan"]["id"] == scan_a_id
    assert retrieved["scan"]["patient_id"] == id_a
    assert retrieved["patient"]["id"] == id_a
    print(f"   [PASS] Retrieved scan {scan_a_id} successfully with associated patient data")

    # 10. Blank Image Safeguard
    print("\n6. Testing Blank Image Safeguard on Patient Scan Upload...")
    blank_img = Image.new("RGB", (100, 100), (255, 255, 255))
    buf_blank = io.BytesIO()
    blank_img.save(buf_blank, format="JPEG")
    buf_blank.seek(0)
    blank_scan_res = requests.post(
        f"{API_BASE}/api/patients/{id_a}/scans",
        files={"file": ("blank.jpg", buf_blank.getvalue(), "image/jpeg")}
    )
    assert blank_scan_res.status_code == 201
    blank_result = blank_scan_res.json()["result"]
    assert blank_result["valid_cervical_image"] is False
    assert "Invalid or unreadable image" in blank_result["message"]
    print(f"   [PASS] Blank image rejected safely with message: '{blank_result['message']}'")

    print("\n>>> ALL PHASE 3 TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    run_tests()
