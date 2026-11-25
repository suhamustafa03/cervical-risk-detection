# Project audit and Vercel readiness

Audit scope: source, configuration, frontend components, API routes, scripts, tests, model artifacts, storage layout, and deployment files. Raw dataset contents, local patient records, and uploaded media were not opened. The existing `docs/test_report.md` and Git stash were left intact.

## Architecture

- **React application:** `frontend/` is a React 19, TypeScript, Vite 8 SPA with Tailwind CSS, Framer Motion, Recharts, Lucide icons, React Router, and Three.js. The primary workflow is implemented in `frontend/src/App.tsx`; `Cervix3DViewer.tsx` loads the bundled `frontend/public/models/reproductive_anatomy.glb` educational model. Styling lives in `App.css` and `index.css`.
- **Python API:** `api.py` exposes FastAPI. `backend_addons.py` adds patient, scan, facility, report, and asset routes. The model checkpoints are `cervical_validator.pth` and `phase2_resnet18_best.pth` (about 43 MB each).
- **Local persistence:** SQLAlchemy uses `cervical_records.db` (SQLite) by default. Original, overlay, heatmap, and report files use local `storage/` directories. The local database and storage are not deployable as durable Vercel state.
- **Vercel target:** root `vercel.json` configures a Vite web service, a Node/Express Blob helper service, and a FastAPI Python service under one Vercel project and URL. The public Production deployment is live at [cervicalproject.vercel.app](https://cervicalproject.vercel.app), backed by a Free Neon PostgreSQL database and private Vercel Blob.
- **Other project material:** root `app.py` and `requirements-frontend.txt` belong to the legacy Streamlit research application. Training/data review scripts, dataset archives, annotations, and local database/media are not runtime dependencies of the hosted site. The existing Render manifest is excluded from Vercel uploads and is not used by this architecture.

## User-facing features and frontend/API traffic

- Patient form validates required names, age, and consent, then calls `POST /api/patients`. The one-time patient capability token is held in app memory and sent in `X-Patient-Access-Token` for protected operations.
- Image selection accepts browser-declared JPG/JPEG/PNG up to 20 MB. Local development submits multipart form data to `POST /api/patients/{patient_id}/scans`. Production obtains `POST /api/patients/{patient_id}/upload-grant`, uploads directly to private Blob through `/api/blob-upload`, then submits the private pathname to the same scan endpoint.
- The result view displays validator outcome, Type 1/2/3 class, confidence, class probabilities, interpretation, next-step wording, original image, and Grad-CAM overlay controls. It labels Grad-CAM as model influence, not lesion segmentation, and states that the prototype is not a clinical diagnosis.
- The PDF report is generated in the browser with jsPDF, downloaded locally, and in production may be saved as a private Blob report associated with the scan. PDF generation is lazy-loaded to keep it out of the initial frontend bundle.
- Facility lookup uses bundled `data/facilities.json`, optionally filtered by city or PIN code and ranked by browser geolocation. It does not call a third-party facility API. Geolocation is requested only after user action.
- The Three.js anatomical view is educational. It does not run inference or mark lesions. About/education content, responsive styling, progress/loading states, and validation/error messages remain in the existing UI.
- Patient/scan history is available from the “View previous cases” dialog using patient ID plus the one-time access code. It lists prior predictions, confidence, interpretation, next-step text, and offers saved report retrieval. The access code is shown on the post-registration screen so the user can retain it; it is not saved in browser storage.

## Backend routes

| Route | Purpose | Access |
| --- | --- | --- |
| `GET /health` | Model-file, database, and storage readiness status | Public status only |
| `POST /analyze` | Standalone multipart image analysis compatibility route | Public; image checks apply |
| `POST /api/patients` | Create consented patient record; return one-time capability token | Public registration |
| `GET /api/patients` | List records | Requires configured `ADMIN_ACCESS_TOKEN`; otherwise 404 |
| `GET /api/patients/{patient_id}` | Retrieve one patient | Patient capability token |
| `GET /api/patients/{patient_id}/scans` | Retrieve that patient's scan history | Patient capability token |
| `POST /api/patients/{patient_id}/upload-grant` | Sign short-lived, path/type/size-specific direct upload | Patient capability token and Blob backend |
| `POST /api/patients/{patient_id}/scans` | Validate, analyze, store, and create scan | Patient capability token and recorded consent |
| `GET /api/scans/{scan_id}` | Retrieve scan and associated patient summary | Patient capability token |
| `GET /api/scans/{scan_id}/assets/{kind}` | Resolve original/overlay/heatmap/report object | Patient capability token |
| `POST /api/scans/{scan_id}/report` | Associate uploaded PDF with scan | Patient capability token |
| `GET /api/scans/{scan_id}/report` | Retrieve a small saved PDF | Patient capability token; larger reports use private asset flow |
| `GET /api/facilities` | Filter/rank bundled facility directory | Public |
| `POST /api/blob-upload` | Verify upload grant and complete Vercel Blob client upload | Signed, expiring upload grant |
| `POST /api/private-asset` | Return short-lived URL after backend authorization | Patient capability token |

## Inference and explainability

- Both models are torchvision ResNet18 architectures with their original state-dict weights. The validator has two outputs; the classifier has three, ordered `Type 1`, `Type 2`, `Type 3`. The actual weights have not been modified.
- Input preprocessing remains RGB, resize to 224×224, tensor conversion, ImageNet mean/std normalization. The validator acceptance threshold remains 0.80; uniform/near-blank low-standard-deviation images are rejected before inference. Vercel uses CPU and lazy model loading.
- Accepted inputs run classifier softmax, select the maximum-probability class, and generate Grad-CAM at `layer4[-1].conv2`. A 48×48 map is returned. Scan storage also creates original, heatmap, and overlay objects when classification is accepted. No validated segmentation mask or lesion metric is produced; persisted quality/metrics fields explicitly indicate not assessed/not computed.
- `GET /health` reports whether checkpoint files and persistent services are configured; models are loaded only on first inference. Errors do not silently substitute mock outputs.

## Data, uploads, and security

- SQL schema: `patients` holds UUID, first/last name, age, gender, case reference, notes, consent, creation time, and capability-token hash. `scans` references patient UUID and holds original/overlay/heatmap/mask/report paths, prediction JSON, predicted type, confidence, quality/metrics JSON, and creation time.
- Schema setup creates missing tables and additively adds `patients.access_token_hash` and `scans.report_path`. `scripts/migrate_database.py` provides an explicit migration entry point. It does not import local rows.
- Production requires hosted PostgreSQL and a **private** Blob store. Original image, overlay, heatmap, and PDF pathnames are stored in PostgreSQL; object bytes are held in Blob. The deployment excludes local `cervical_records.db*`, `storage/`, and raw `data/*` while including only `data/facilities.json` and both model checkpoints.
- Upload checks enforce JPG/PNG and PDF MIME allowlists as applicable, actual decoded image type, a 20 MB cap, and 20 megapixels. Images are decoded and re-encoded before being linked as originals to strip metadata. Production image uploads go browser-to-Blob, avoiding Vercel's 4.5 MB function request-body cap; the inference call carries only a small Blob pathname.
- Patient/scan/report/asset reads require a high-entropy capability token; only its SHA-256 hash is stored. The all-patient list is disabled unless a separate admin token is configured. Blob reads use private access and short-lived signed URLs. Secrets are environment-only and `.env` files are ignored.
- There is no full user login/identity system, rate limiting, or clinic authorization workflow. A public deployment would allow the public to create new records, so public demonstrations must use synthetic records and images. No regulatory compliance or clinical validation is asserted.
- No external AI service or API key is integrated. Facility data is local/static; geolocation is provided by the browser. Google Fonts are an external frontend resource.

## Findings and deployment blockers

| Finding in the original/local setup | Vercel consequence | Current disposition |
| --- | --- | --- |
| SQLite and local files were the default persistence model | Function filesystem is ephemeral and shared state is not guaranteed | Local mode retained; production requires PostgreSQL + private Blob and refuses ephemeral SQLite fallback |
| Multipart upload could exceed the Vercel 4.5 MB body limit | A 20 MB image would fail before inference | Production client-to-Blob direct upload and pathname handoff implemented |
| Frontend had a localhost API fallback | Production browser requests would target the visitor's own machine | Same-domain production API base; local Vite proxy preserved |
| Patient and scan routes had no record-level authorization | Guessed IDs could disclose records/assets | One-time capability tokens, admin-only list, signed uploads and private assets added |
| Two PyTorch checkpoints plus Linux dependencies exceed standard function bundle size | Standard 500 MB bundle is insufficient | Estimated extracted bundle is ~999.4 MiB; Large Functions (up to 5 GB) and Fluid Compute are required |
| No persistent PostgreSQL database was available for the deployment | Patient creation and saved scan history cannot work in production without a database | Neon Free (`free_v3`) is connected to Production, schema migrated, and patient/scan tables verified empty after synthetic cleanup |
| No Git remote was configured | Git-connected deploy cannot be initiated from this checkout | Vercel CLI deployment is the available route after login, or connect a repo explicitly |
| Existing patient/scan history API was not exposed as a frontend history screen | Users could not retrieve prior cases from the UI | Token-protected history dialog now calls the API and supports saved report retrieval |

## Dependency and build review

- Python runtime dependencies are pinned in root `requirements.txt`; CPU-only Linux PyTorch/Torchvision wheels are selected for Vercel. `.python-version` pins Python 3.12. `requirements-backend.txt` refers to the same pins. `requirements-frontend.txt` is for the separate legacy Streamlit app.
- Linux CPython 3.12 wheel extraction plus both checkpoints and API/facility files was measured at approximately 999.4 MiB before platform overhead. Vercel's first remote function measurement was 1000.08 MB; the standard 500 MB limit failed, then Preview built successfully after enabling Large Functions. Production remains undeployed.
- Frontend dependencies and lockfile are in `frontend/package.json` and `frontend/package-lock.json`. PDF and Three.js viewer code are lazy-loaded. The viewer chunk still emits Vite's >500 KB chunk advisory, but it is loaded only when the viewer is opened.
- `render.yaml` is not referenced by Vercel and is excluded by `.vercelignore`. Root deployment routing is centralized in `vercel.json`; a service-level Vite rewrite serves the SPA for nested-page refreshes.

## Verification performed

- `npm run build`: passed; TypeScript and production Vite build completed.
- `npm run lint`: passed.
- Synthetic backend suite `tests/test_vercel_deployment.py`: 7 tests passed using temporary database/media directories. It exercised health/configuration, malformed MIME and image rejection, actual checkpoint lazy load and screening rejection, classifier/Grad-CAM hooks on a deterministic synthetic tensor, patient/scan authorization and history isolation, PDF persistence/retrieval, and facility filtering.
- Local Vercel build recognized all three services. Production is deployed and public at [cervicalproject.vercel.app](https://cervicalproject.vercel.app); Preview remains READY behind Vercel Authentication at [cervicalproject-e45d7i2yb-suhamustafa03s-projects.vercel.app](https://cervicalproject-e45d7i2yb-suhamustafa03s-projects.vercel.app). Build logs show the Vite frontend, Express helper service, and Python backend; deployment metadata reports Node.js and Python runtimes.
- Production checks on 2026-10-30 returned `200` for `/`, `/cases/history`, `/health`, and `/api/facilities?city=Delhi`; GET `/api/blob-upload` returned the expected `405`. Health reported `status=ok`, model checkpoint files present, CPU execution, Blob storage selected, and a configured database. `models_loaded=false` is expected until a model request warms that function instance. The downloaded public app bundle contains no localhost API URL, and the live page produced no browser console errors.
- Neon Marketplace terms were accepted and a Free Neon (`free_v3`) database was provisioned in `iad1`, attached to Production only, and migrated without importing local data. A synthetic registration/auth check against this database passed: create returned `201`, missing capability token returned `401`, invalid token returned `403`, authorized retrieval returned `200`, and history returned `200`. The temporary synthetic row was deleted; final patient and scan counts were both zero.
- A generated noise PNG was posted to the Preview `/analyze` endpoint; the live validator loaded the shipped model and rejected the synthetic input with the expected non-cervical response. It did not create a record or Blob object. The hosted classifier/Grad-CAM positive path remains unverified with live input because no valid synthetic cervical image exists; local tests exercised classifier and Grad-CAM with synthetic tensors. No real patient image was uploaded.
- The first remote build failed at 1000.08 MB against the standard 500 MB Python limit. Preview-only `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` allowed the retry to complete. The account remains on Hobby; no paid upgrade was made.
- A private Vercel Blob store is linked to Preview and Production. A synthetic live upload attempt created one private object, but the Node smoke client later failed with `fetch failed`, so the full image-analysis/history/report sequence did not complete. The marked synthetic patient, any associated scan, and the object were removed; cleanup verified zero patient and scan rows. No real patient image was uploaded.
- Browser checks covered the live registration page, which rendered without console errors, and a 390px viewport with no horizontal overflow; earlier local browser checks covered facility lookup and nested SPA route refresh. A complete positive production prediction and live report retrieval remain unverified. The live deployment must not be described as having passed a full image-to-report test.
- The scan-history dialog and one-time access-code display were added after the initial browser smoke check. TypeScript build and ESLint pass; interactive browser testing of that new dialog remains part of live-site smoke checks.
- The legacy `tests/run_e2e_tests.py` reads from `storage/originals/` and overwrites the existing `docs/test_report.md`; `tests/test_patient_flow.py` assumes an unauthenticated live server and writes records. They were not run against user data. The new isolated suite is the deployment test path.
- Production is live and the hosted PostgreSQL migration and synthetic patient authorization flow were verified. The checkout has no Git remote, so deployment used the authenticated Vercel CLI. Full live image-to-report testing is still pending because the Node synthetic smoke client encountered a fetch failure; no paid upgrade or real patient data was used.

## Deployment checklist

- [x] Sign in to Vercel CLI, create the project, and deploy a protected Preview.
- [x] Deploy the Large Functions bundle on Hobby in Preview.
- [x] Deploy and verify all three Vercel services and nested SPA refresh in Preview.
- [x] Create an empty private Blob store on the Hobby free allowance.
- [x] Accept Neon Marketplace terms and provision Neon on the explicit Free plan.
- [x] Configure Production PostgreSQL, private Blob storage, storage backend selection, and upload signing secret without committing credentials.
- [x] Migrate the empty hosted database; verify synthetic authorization and remove all synthetic rows and objects.
- [x] Build and deploy the three Vercel services from the repository root using the Vercel CLI and Large Functions on Hobby.
- [x] Verify public Production health, SPA and nested refresh, facility lookup, and helper method routing.
- [ ] Complete synthetic live browser upload-to-prediction, private-asset, report-retrieval, and responsive checks across the result/history views. The home page passes at 390px width. No real patient data; stop if Vercel asks for a paid upgrade.
