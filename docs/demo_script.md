# Live Demonstration Script & Presentation Guide

**Project:** Early Cervical Risk Detection using Explainable Colposcopy Analysis  
**Audience:** Technical evaluators, clinical researchers, and healthcare technology reviewers  
**Prototype URL:** [http://localhost:5173](http://localhost:5173)  
**Backend API:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  

---

## 1. Demonstration Setup (Pre-Flight Checklist)

Before commencing the demonstration, ensure both services are running:
1. **Backend Daemon:** `uvicorn api:app --host 127.0.0.1 --port 8000` (port 8000)
2. **Frontend Dev Server:** `npm run dev` in `frontend/` (port 5173)
3. **Run Setup Script:**
   ```bash
   python3 scripts/setup_demo.py
   ```
4. **Sample High-Confidence Image Asset:** Located at `data/demo_sample.jpg` (Type 1, 99.9% confidence).

---

## 2. Step-by-Step Live Walkthrough

### Step 1: Opening & Patient Registration (Step 1 of 3)

1. Navigate to `http://localhost:5173` in Safari, Chrome, or Edge.
2. **Presenter Script / Talking Points:**
   > *"Welcome. This system is a research prototype designed to investigate deep learning for colposcopy image quality screening, transformation-zone classification, and explainable AI visualizations."*
   > *"Notice that data privacy and consent are enforced at the architectural level. Registration requires explicit patient consent before any diagnostic pixels are transferred."*
3. **Actions to perform:**
   - First Name: `Ananya`
   - Last Name: `Sharma`
   - Age: `32`
   - Case Reference: `REF-2026-DEMO`
   - Check the **"Patient consent confirmed"** checkbox.
   - Click **"Continue to image upload"**.

---

### Step 2: Input Screening & Image Safeguards (Step 2 of 3)

1. **Presenter Script / Talking Points:**
   > *"Before attempting deep-learning inference, the system passes incoming images through a two-tier screening safeguard: a pixel-variance quality check to eliminate blank or dark captures, and a binary ResNet-18 validator to reject non-cervical photographs."*
2. **Actions to perform (Optional Negative Control Test):**
   - If showcasing the safeguard, upload a blank image or non-cervical photo (e.g. `keychain_test.jpg`).
   - Observe immediate rejection: *"The input screening model did not classify this image as cervical. Try a suitable colposcopy image."*
3. **Actions for Happy Path Demonstration:**
   - Upload `data/demo_sample.jpg`.
   - The preview renders instantly with file details.
   - Click **"Analyze image"**.

---

### Step 3: Clinical Analysis Panel & Grad-CAM Explanation (Step 3 of 3)

1. **Presenter Script / Talking Points:**
   > *"The analysis completes synchronously in under a second. The system presents four interconnected components:"*
   - **Transformation Zone (TZ) Type:** Classified into Type 1, 2, or 3 based on squamocolumnar junction visibility.
   - **Confidence & Class Probabilities:** Fully calibrated probability bars with 1-decimal consistency.
   - **Grad-CAM Visualization:** An attribution heatmap overlay generated from the model's final convolutional layer (`conv2` of `layer4`), highlighting the anatomical regions that drove the prediction.
   - **Attention Color Scale:** Clinicians can adjust overlay opacity from 0% to 100% or toggle to the original image, guided by the scientific color scale.
2. **Actions to perform:**
   - Slide the **Heatmap Opacity** slider to show how the heatmap aligns over the squamocolumnar junction.
   - Point out the **Clinical Analysis** explanation: *"The transformation zone is completely visible on the ectocervix."*
   - Highlight the **Lesion Assessment Limitation Card**:
     > *"Dedicated lesion detection is not available in this prototype. A clean-looking image or the absence of highlighted regions does not confirm that the cervix is lesion-free."*

---

### Step 4: 3D Anatomical Spatial Context

1. **Presenter Script / Talking Points:**
   > *"To assist patient communication and anatomical understanding, the interface features a 3D anatomical viewer built with Three.js and WebGL."*
2. **Actions to perform:**
   - Rotate and zoom the 3D reproductive anatomy model to demonstrate interactive spatial context.

---

### Step 5: Downloadable Clinical PDF Report (Marvix AI Medical Layout)

1. **Presenter Script / Talking Points:**
   > *"Clinicians need portable, standardized documentation. We redesigned the PDF export to match the modern Marvix AI clinical layout in a professional teal palette."*
2. **Actions to perform:**
   - Click **"Download clinical report (PDF)"**.
   - Open the generated PDF in the system viewer.
   - **Highlight Key Sections:**
     1. Teal header banner with unique Report & Patient ID.
     2. Numbered Section `1` (Patient Demographics table with dividers).
     3. Numbered Section `2` (Clinical Classification & Confidence score).
     4. Numbered Section `3` (Centered Grad-CAM visualization composited offscreen with exact aspect ratio).
     5. Numbered Section `4` (Recommendations & Lesion Limitation disclaimer).
     6. Institutional research disclaimer in the footer.

---

### Step 6: Nearby Referral & Care Centres

1. **Presenter Script / Talking Points:**
   > *"When follow-up colposcopy or biopsy is indicated, clinicians and patients need immediate access to verified healthcare facilities. The system includes an integrated Care Centres lookup connecting users to public cancer institutes and tertiary gynaecology centres."*
2. **Actions to perform:**
   - Click **"Find nearby referral centres"** (or **"Nearby care centres"** in the sidebar).
   - Enter `Delhi` or `Mumbai` in the search bar, or click **"Near me"** to trigger coordinate proximity sorting.
   - Observe verified cards showing facility name, type, distance in km, phone contacts, and colposcopy specialities.

---

## 3. Frequently Asked Evaluator Questions

**Q: Does this model detect or segment cervical cancer lesions?**  
*A: No. As documented in `docs/lesion_segmentation_findings.md`, the model classifies transformation zone visibility (Type 1/2/3) and provides Grad-CAM attribution. It does not perform lesion segmentation, contouring, or depth measurements. Dedicated lesion assessment requires multi-observer annotated datasets and a dedicated segmentation architecture (e.g. U-Net).*

**Q: Are Patient records isolated?**  
*A: Yes. All scans are bound by foreign key to `patient_id` in SQLite. Automated test suite `tests/test_patient_flow.py` confirms that querying Patient A's history returns zero records from Patient B.*

**Q: What happens if an invalid or blurry image is uploaded?**  
*A: The API runs image quality checks (rejecting blank or standard deviation < 10 images) and binary screening (ResNet-18 validator) before classification, preventing garbage-in/garbage-out results.*
