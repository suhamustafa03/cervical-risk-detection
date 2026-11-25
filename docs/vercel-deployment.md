# Vercel deployment and maintenance

## Architecture

The root [`vercel.json`](../vercel.json) defines one Vercel project with three services on the same deployment URL:

- `web`: the existing React, Vite, and TypeScript app in `frontend/`.
- `helpers`: a small Node.js/Express service in `vercel-helpers/` for the Vercel Blob client-token and private-asset URL routes.
- `backend`: the existing FastAPI app at `api:app` in the repository root.
- `/api/blob-upload` and `/api/private-asset` route to `helpers`; other `/api/*`, `/health`, and `/analyze` requests route to FastAPI. All other paths route to the Vite SPA. A service-level rewrite serves `index.html` for nested-page refreshes.
- Vercel service binding `BACKEND_INTERNAL_URL` lets the private asset route ask FastAPI for a patient-authorized Blob pathname. The Node function returns a five-minute, single-object private Blob URL.

Vercel Services must be enabled for the project in **Settings → Build and Deployment → Framework Preset: Services**. Services are in beta on all plans and share a public Vercel deployment URL. The backend function explicitly includes both model checkpoints and the facility JSON, excludes local/test/frontend data, and allows up to 300 seconds per request. No Render service is part of this configuration.

## Current Vercel status (2026-10-30)

- Project `cervical_project` is on the Vercel Hobby plan. Production is deployed and publicly reachable at [cervicalproject.vercel.app](https://cervicalproject.vercel.app). The Preview remains protected by Vercel Authentication at [cervicalproject-e45d7i2yb-suhamustafa03s-projects.vercel.app](https://cervicalproject-e45d7i2yb-suhamustafa03s-projects.vercel.app).
- The production build completed with Node 24, Vite, Express helpers, and the Python 3.12 backend. Vercel reports Node.js and Python runtimes. The function bundle is about 1000 MB versus the standard 500 MB limit; `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` enables Large Functions, and the build completed without a paid plan upgrade.
- A private Vercel Blob store, `cervical-project-private`, is in `iad1` and connected to Preview and Production. No user or real patient image/report data is stored there. Hobby storage/transfer quotas apply.
- The Neon Marketplace terms were accepted. Database `cervical-project-db` is on the explicit Free plan (`free_v3`), in `iad1`, and connected to Production only. The schema was migrated without importing local SQLite data. `patients` and `scans` were verified empty after a synthetic smoke-test cleanup.
- Production environment variables configure PostgreSQL, private Blob storage, `CERVICAL_STORAGE_BACKEND=blob`, and the matching upload signing secret for the helper and backend services. The all-patient admin token is not configured, so that route remains unavailable.
- Read-only Production checks on 2026-10-30 returned `200` for `/`, a nested SPA path, `/health`, and `/api/facilities?city=Delhi`; `/api/blob-upload` correctly returned `405` for GET. `/health` reported `status=ok`, CPU, both checkpoint files present, Blob storage selected, and a configured database. Models load lazily, so `models_loaded=false` at health-check time is expected.
- A synthetic live upload attempt created a marked test patient and a private image object. The Node smoke client then failed with a network `fetch failed` before the image/report sequence could be fully verified. Only that synthetic patient, any scan with its marked ID, and its object were removed; cleanup verified zero patient and scan rows. No real patient data was used. Direct browser-to-Blob upload, live positive prediction/Grad-CAM, and saved-report retrieval remain unverified end-to-end.

## Runtime and model limits

The backend uses Python 3.12 and pinned CPU-only PyTorch 2.7.1 / Torchvision 0.22.1 wheels. Both original state-dict checkpoints stay in the repo and load lazily on first inference. Model architecture, class order, transforms, thresholds, and Grad-CAM target layer are retained. Vercel Functions run the model on CPU; `torch.set_num_threads(1)` limits per-instance CPU contention.

I measured the Linux CPython 3.12 x86_64 wheels and both checkpoints. The extracted pinned dependencies total about **913.9 MiB**; adding both 43 MiB checkpoints and the API/facility files brings the estimate to **about 999.4 MiB**, before Vercel runtime overhead. The first remote build confirmed a 1000.08 MB function against the standard 500 MB limit. Production built successfully with Vercel **Large Functions** enabled through `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`; this was a deployment setting, not a plan upgrade. No prediction is mocked or substituted.

`/analyze` keeps its multipart API for local development. Production uploads use a signed, private Blob client upload first, then submit only a small JSON pathname to FastAPI. This keeps the 20 MB image out of Vercel's 4.5 MB Function request/response body path. The backend then reads the private object, checks its actual image format, dimensions and size, strips image metadata when saving the original, and performs both trained-model inference and Grad-CAM. The production deployment has not yet passed the full live upload-to-report flow; see the status above.

## Persistent records and files

- **Database:** configure a managed PostgreSQL `DATABASE_URL` through the Vercel Marketplace (Neon is the supported setup documented for this project). The ORM preserves the patient-to-scan relationship and stores prediction JSON, confidence, status metadata, timestamps, private object pathnames, and the report pathname. Existing SQLite remains the local default.
- **Migration:** run `DATABASE_URL='<target connection string>' python scripts/migrate_database.py` for an explicit additive migration. The app also applies the same additive column migration at startup. It adds `patients.access_token_hash` and `scans.report_path`; it does not copy, read, or upload the local database.
- **Images and reports:** create a **private** Vercel Blob store. Original image, heatmap, overlay, and PDF objects stay private. The original uploaded object is decoded, sanitized, and overwritten with its metadata-stripped copy before it is linked to a scan. No uploaded content is stored in a Vercel Function's temporary filesystem.
- **Report generation:** the existing browser-generated jsPDF report still downloads to the user's device. The UI also uploads and links a private persistent copy. `GET /api/scans/{scan_id}/report` returns smaller PDFs; for any stored object, `POST /api/private-asset` returns an expiring private URL after FastAPI checks the patient capability token.
- **No data migration:** never import `cervical_records.db`, `storage/`, local image files, or patient records into the public deployment. Use synthetic records and images only for public testing.

The current Vercel Blob Hobby allocation is 1 GB average storage, 10 GB monthly Blob transfer, 10,000 simple operations, and 2,000 advanced operations. Hobby Blob stops serving after its included limits rather than charging for overage. Neon currently documents 1 GB storage and 100 compute-hours per Free project each month. Vercel services also consume Function CPU, memory, invocation, and transfer allowances. Check the live project dashboard before enabling anything paid; this project does not provision a paid plan or enable paid overages.

## Security controls and limits

- Patient creation returns a high-entropy capability token once; only its SHA-256 hash is stored. Patient, scan, report, and asset endpoints require `X-Patient-Access-Token`. The patient ID alone does not authorize access.
- The legacy all-patient listing route requires a separate `ADMIN_ACCESS_TOKEN`; it returns 404 if no admin token is configured.
- Direct Blob upload grants are HMAC-signed, expire after 15 minutes, bind one patient/scan/path/content type, allow JPG/PNG or PDF only, and cap uploads at 20 MB. Configure the same `UPLOAD_SIGNING_SECRET` on the helpers and backend services.
- Blob storage must be created as **private**. `BLOB_READ_WRITE_TOKEN` stays in server environment variables. It is never compiled into the frontend. Python Blob access uses Vercel OIDC with `BLOB_STORE_ID` when available, or a server-side read/write token.
- Same-domain production requests do not need CORS. Vercel deployments disable the broad `*.vercel.app` CORS regex. Local Vite development uses its proxy to `127.0.0.1:8000`.
- There is no user account/identity system in the original project. Capability tokens protect individual records, but anyone who can use the public app can create a new record. Public demonstrations must use synthetic data. This project makes no claim of regulatory or clinical validation.
- Vercel logs should not include image bytes, tokens, database URLs, or patient payloads. Keep secrets out of `.env.example`; only the examples are version-controlled.

## Required Vercel environment variables

Set variables in the Vercel project and apply them to **Production, Preview, and Development** only where appropriate:

| Name | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | Yes | PostgreSQL URL from the connected Marketplace database. Use its pooled URL for runtime connections. |
| `CERVICAL_STORAGE_BACKEND` | Yes | Set to `blob` in Vercel. |
| `BLOB_READ_WRITE_TOKEN` | Yes for browser uploads | Server-only Blob credential injected by the private Vercel Blob store. |
| `BLOB_STORE_ID` | Required for OIDC | Blob store ID used by Vercel's Python SDK OIDC flow; the integration may provide it automatically. |
| `UPLOAD_SIGNING_SECRET` | Yes | At least 32 random characters; use the same secret for the helpers and backend services. |
| `VERCEL_SUPPORT_LARGE_FUNCTIONS` | If not auto-enabled | Set to `1` for a project that needs the Large Functions bundle tier. |
| `ADMIN_ACCESS_TOKEN` | Optional | Separate secret for the existing all-patient API list route. Do not add it to the frontend. |
| `VITE_API_BASE_URL` | No | Leave empty; production uses same-domain paths. Never set this to localhost. |

`ALLOWED_ORIGINS` is only for local cross-origin development. `CERVICAL_STORAGE_DIR` is local-only. Vercel's `VERCEL_*` deployment values are platform-managed.

## Deployment checklist

1. Keep Neon on its Free plan (`free_v3`). No paid service or upgrade is enabled.
2. Keep `DATABASE_URL` Production-only; do not connect this production database to Preview or Development.
3. Keep `UPLOAD_SIGNING_SECRET` aligned between Helpers and Backend and outside Git. The private Blob store is attached to Preview and Production.
4. For a replacement empty database, run `python scripts/migrate_database.py` against that hosted database only. Never point it at the local patient database.
5. Keep Large Functions enabled. If Vercel requests a plan upgrade or payment, stop and get explicit approval.
6. The current checkout has no Git remote; production deployment was performed with the authenticated Vercel CLI.
7. Complete the remaining synthetic live checks: browser direct upload, invalid-image rejection, model inference on an appropriate synthetic fixture if available, private asset retrieval, saved report retrieval, and responsive layout. Do not use real patient data.

## Local development

Run FastAPI on port 8000 (`python api.py`) and run `cd frontend && npm run dev`. Vite proxies `/api`, `/analyze`, and `/health` to FastAPI. Local SQLite and local `storage/` remain the defaults. Do not point a Vercel deployment at those local paths.

For isolated synthetic backend checks, install `requirements-test.txt` in a virtual environment and run `python -m unittest discover -s tests -p 'test_vercel_deployment.py'`. The suite creates a temporary database and media directory, loads the actual checkpoints, and never reads `storage/`, `dataset/`, or writes `docs/test_report.md`. The older `tests/run_e2e_tests.py` is intentionally not used for deployment smoke tests because it reads `storage/originals/` and overwrites the existing test report; `tests/test_patient_flow.py` assumes an unauthenticated live server and is also legacy.

The standalone `requirements-frontend.txt` belongs to the legacy Streamlit research app; the deployed React frontend uses `frontend/package.json` and `frontend/package-lock.json`.
