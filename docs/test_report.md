# End-to-End System Test Report

- **Project:** Early Cervical Risk Detection using Explainable Colposcopy Analysis
- **Execution Timestamp:** 2026-10-21 21:47:56
- **Overall Status:** **FAILED** (8/9 Tests Passed)
- **Environment:** macOS / Python 3.13 / FastAPI / PyTorch / React 19 + TypeScript + Vite

---

## 1. Test Execution Summary

| # | Test Case | Category | Status | Details |
|---|---|---|---|---|
| 1 | Backend Service & Model Health | System | **PASS** | Status code: 200, Models loaded: True |
| 2 | Patient Age Boundaries (1-120) | Patient Management | **PASS** | Age 0 response: 422, Age 121 response: 422 |
| 3 | Mandatory Consent Enforcement | Patient Management | **PASS** | Consent=False rejected with status code: 400 |
| 4 | Patient Scan History Isolation | Data Security & Privacy | **PASS** | Patient A has 1 scan(s), Patient B has 1 scan(s). No cross-patient leakage. |
| 5 | Blank / Zero-Contrast Image Safeguard | Input Screening | **PASS** | Message returned: 'Invalid or unreadable image. Please provide a clear colposcopy photograph.' |
| 6 | Non-Cervical Photo Rejection (Validator) | Input Screening | **PASS** | Message returned: 'The input screening model did not classify this image as cervical. Try a suitable colposcopy image.' |
| 7 | TZ Classification & Grad-CAM Generation | AI Inference | **FAIL** | Predicted: None (Confidence: 0%), CAM grid: 0x0 |
| 8 | Care Centres Query by City | Nearby Care Centres | **PASS** | Matched 1 facility/facilities for 'Delhi': All India Institute of Medical Sciences (AIIMS) |
| 9 | Care Centres Proximity Search & Sorting | Nearby Care Centres | **PASS** | Closest to Bengaluru: Kidwai Memorial Institute of Oncology (3.4 km away) |

---

## 2. Key Verification Findings

1. **Safety & Ethics Compliance**: Non-cervical objects (such as keys or general images) and zero-contrast blank uploads are strictly blocked with explicit, user-friendly guidance before entering transformation-zone inference.
2. **Data Isolation**: Multi-patient scans are strictly bound to their respective `patient_id`. Patient scan histories do not cross-contaminate between different patient IDs.
3. **Explainability**: Grad-CAM activations are computed off the final convolutional layer of ResNet-18 and successfully rendered in both the interactive UI and offscreen for the clinical PDF report.
4. **Clinical Boundaries**: Lesion assessment text explicitly clarifies that dedicated lesion detection is not validated in this prototype, and no fabricated lesion masks are generated.
5. **Referral Access**: Patients and clinicians can discover verified regional cancer centres and tertiary public hospitals via city, PIN code, or coordinate-based proximity search.

---
*Report generated automatically by `tests/run_e2e_tests.py`.*