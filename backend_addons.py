"""Patient, scan, and persistent-storage routes for the FastAPI application.

Local development keeps the existing SQLite and ``storage/`` defaults. Vercel
deployments require PostgreSQL and private Vercel Blob; ephemeral function disks
are never used for patient images or reports.
"""
from __future__ import annotations

import hashlib
import hmac
import base64
import json
import os
import secrets
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from starlette.datastructures import UploadFile as StarletteUploadFile

BASE_DIR = Path(__file__).resolve().parent
IS_VERCEL = os.getenv("VERCEL", "").lower() in {"1", "true"} or bool(os.getenv("VERCEL_ENV"))
STORAGE_BACKEND = os.getenv(
    "CERVICAL_STORAGE_BACKEND", "blob" if IS_VERCEL else "local"
).lower()
STORAGE_DIR = Path(os.getenv("CERVICAL_STORAGE_DIR", str(BASE_DIR / "storage"))).resolve()
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_REPORT_BYTES = 20 * 1024 * 1024

if STORAGE_BACKEND == "local":
    for folder in ("originals", "masks", "overlays", "heatmaps", "reports"):
        (STORAGE_DIR / folder).mkdir(parents=True, exist_ok=True)

_configured_database_url = os.getenv("DATABASE_URL")
if IS_VERCEL and not _configured_database_url:
    # Never silently fall back to a non-persistent SQLite file on Vercel.
    DATABASE_URL = ""
    engine = None
    SessionLocal = None
else:
    DATABASE_URL = _configured_database_url or f"sqlite:///{(BASE_DIR / 'cervical_records.db').as_posix()}"
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    if DATABASE_URL.startswith("sqlite:///") and not DATABASE_URL.startswith("sqlite:///:memory:"):
        sqlite_file = Path(DATABASE_URL.replace("sqlite://", "", 1))
        sqlite_file.parent.mkdir(parents=True, exist_ok=True)
    engine_kwargs: dict[str, Any] = {}
    if DATABASE_URL.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        engine_kwargs.update(pool_pre_ping=True, pool_recycle=300, pool_size=1, max_overflow=0)
    engine = create_engine(DATABASE_URL, future=True, **engine_kwargs)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[str] = mapped_column(String(30), nullable=False)
    case_ref: Mapped[str | None] = mapped_column(String(100), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    consent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    access_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    scans: Mapped[list["Scan"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False, index=True)
    original_path: Mapped[str] = mapped_column(Text, nullable=False)
    overlay_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    heatmap_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    mask_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    prediction_json: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    patient: Mapped[Patient] = relationship(back_populates="scans")


def migrate_schema() -> None:
    """Create missing tables and apply additive fields without touching records."""
    if engine is None:
        return
    Base.metadata.create_all(bind=engine)
    columns = {table: {col["name"] for col in inspect(engine).get_columns(table)}
               for table in ("patients", "scans")}
    additions = {
        "patients": {"access_token_hash": "VARCHAR(64)"},
        "scans": {"report_path": "TEXT"},
    }
    with engine.begin() as connection:
        for table, fields in additions.items():
            for column, sql_type in fields.items():
                if column not in columns[table]:
                    if engine.dialect.name == "postgresql":
                        connection.execute(text(
                            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {sql_type}"
                        ))
                    else:
                        connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))


migrate_schema()


class PatientCreate(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=1, le=120)
    gender: str
    case_ref: str | None = Field(default=None, max_length=100)
    notes: str | None = Field(default=None, max_length=5000)
    consent: bool


class UploadGrantRequest(BaseModel):
    scan_id: str
    content_type: str
    kind: str = "image"


class BlobScanCreate(BaseModel):
    scan_id: str
    blob_pathname: str
    content_type: str


def _require_database() -> None:
    if engine is None or SessionLocal is None:
        raise HTTPException(status_code=503, detail="Persistent database is not configured.")


def _patient_dict(patient: Patient) -> dict[str, Any]:
    return {
        "id": patient.id,
        "first_name": patient.first_name,
        "last_name": patient.last_name,
        "name": f"{patient.first_name} {patient.last_name}",
        "age": patient.age,
        "gender": patient.gender,
        "case_ref": patient.case_ref,
        "notes": patient.notes,
        "consent": patient.consent,
        "created_at": patient.created_at.isoformat() if patient.created_at else None,
    }


def _path_exists(path: str | None) -> bool:
    if not path:
        return False
    if path.startswith("blob://"):
        return True
    return Path(path).is_file()


def _scan_dict(scan: Scan) -> dict[str, Any]:
    try:
        prediction = json.loads(scan.prediction_json)
    except (TypeError, json.JSONDecodeError):
        prediction = {}
    return {
        "id": scan.id,
        "patient_id": scan.patient_id,
        "predicted_type": scan.predicted_type,
        "confidence": scan.confidence,
        "prediction": prediction,
        "created_at": scan.created_at.isoformat() if scan.created_at else None,
        "has_original": _path_exists(scan.original_path),
        "has_overlay": _path_exists(scan.overlay_path),
        "has_heatmap": _path_exists(scan.heatmap_path),
        "has_report": _path_exists(scan.report_path),
    }


def _get_patient(db, patient_id: str) -> Patient | None:
    return db.query(Patient).filter(Patient.id == str(patient_id).strip()).first()


def _get_scan(db, scan_id: str) -> Scan | None:
    return db.get(Scan, str(scan_id).strip())


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _authorize_patient(patient: Patient, token: str | None) -> None:
    if not patient.access_token_hash and not IS_VERCEL:
        # Existing local SQLite rows predate capability tokens; retain local
        # development access until their owners choose to recreate them.
        return
    if not token or not patient.access_token_hash:
        raise HTTPException(status_code=401, detail="Patient access token is required.")
    if not hmac.compare_digest(patient.access_token_hash, _token_hash(token)):
        raise HTTPException(status_code=403, detail="Patient access token is invalid.")


def _upload_secret() -> bytes:
    secret = os.getenv("UPLOAD_SIGNING_SECRET", "")
    if len(secret) < 32:
        raise HTTPException(status_code=503, detail="Secure upload signing is not configured.")
    return secret.encode("utf-8")


def _signed_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).rstrip(b"=")
    signature = hmac.new(_upload_secret(), encoded, hashlib.sha256).digest()
    return encoded.decode("ascii") + "." + base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")


def _mime_for_extension(extension: str) -> str:
    return "image/png" if extension == ".png" else "image/jpeg"


def _validate_image(raw: bytes, content_type: str, image_module) -> Any:
    if content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPG or PNG image.")
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is empty or exceeds the 20 MB limit.")
    try:
        with image_module.open(BytesIO(raw)) as probe:
            actual_format = probe.format
            if probe.width * probe.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions exceed the 20 megapixel limit.")
            probe.verify()
        expected_format = "PNG" if content_type == "image/png" else "JPEG"
        if actual_format != expected_format:
            raise HTTPException(status_code=415, detail="Image content does not match its declared JPG or PNG format.")
        with image_module.open(BytesIO(raw)) as decoded:
            decoded.load()
            return decoded.convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable JPG or PNG image.") from exc


def _blob_sdk():
    try:
        from vercel import blob
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Vercel Blob SDK is not installed.") from exc
    return blob


async def _read_blob(pathname: str, *, max_bytes: int = MAX_IMAGE_BYTES) -> tuple[bytes, str]:
    if STORAGE_BACKEND != "blob":
        raise HTTPException(status_code=404, detail="Blob storage is not enabled.")
    try:
        result = await _blob_sdk().get_blob(pathname, access="private")
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Could not retrieve the private uploaded image.") from exc
    body = result.body
    if len(body) > max_bytes:
        raise HTTPException(status_code=413, detail="Stored file exceeds its allowed size.")
    return body, result.metadata.content_type or "application/octet-stream"


async def _store_asset(category: str, patient_id: str, scan_id: str, filename: str,
                       raw: bytes, content_type: str) -> str:
    if STORAGE_BACKEND == "blob":
        pathname = f"patients/{patient_id}/scans/{scan_id}/{category}.{filename.rsplit('.', 1)[-1]}"
        try:
            await _blob_sdk().put_blob(
                pathname,
                raw,
                access="private",
                content_type=content_type,
                add_random_suffix=False,
                allow_overwrite=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Could not save the image to private storage.") from exc
        return f"blob://{pathname}"
    destination = STORAGE_DIR / category / filename
    destination.write_bytes(raw)
    return str(destination)


async def _replace_private_blob(pathname: str, raw: bytes, content_type: str) -> None:
    try:
        await _blob_sdk().put_blob(
            pathname,
            raw,
            access="private",
            content_type=content_type,
            add_random_suffix=False,
            allow_overwrite=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Could not sanitize the uploaded image in private storage.") from exc


async def _load_blob(path: str) -> tuple[bytes, str]:
    pathname = path.removeprefix("blob://")
    return await _read_blob(pathname)


async def _request_image(request: Request, model_api, patient_id: str, *, allow_blob: bool = True):
    content_type_header = request.headers.get("content-type", "").lower()
    if content_type_header.startswith("multipart/form-data"):
        form = await request.form()
        file = form.get("file")
        if not isinstance(file, StarletteUploadFile):
            raise HTTPException(status_code=422, detail="Image field 'file' is required.")
        try:
            content_type = file.content_type or ""
            raw = await file.read(MAX_IMAGE_BYTES + 1)
            image = _validate_image(raw, content_type, model_api.Image)
            return raw, content_type, image, None, None
        finally:
            await file.close()

    if content_type_header.startswith("application/json") and allow_blob:
        try:
            body = await request.json()
            payload = BlobScanCreate.model_validate(body)
        except Exception as exc:
            raise HTTPException(status_code=422, detail="A valid private blob upload reference is required.") from exc
        try:
            scan_id = str(uuid.UUID(payload.scan_id))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid scan ID.") from exc
        if payload.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(status_code=415, detail="Upload a JPG or PNG image.")
        ext = ".png" if payload.content_type == "image/png" else ".jpg"
        expected_path = f"patients/{patient_id}/scans/{scan_id}/original{ext}"
        if payload.blob_pathname != expected_path:
            raise HTTPException(status_code=400, detail="Uploaded image path is invalid.")
        raw, stored_content_type = await _read_blob(payload.blob_pathname)
        if stored_content_type.split(";", 1)[0].strip().lower() != payload.content_type:
            raise HTTPException(status_code=415, detail="Stored image content type does not match the request.")
        image = _validate_image(raw, payload.content_type, model_api.Image)
        return raw, payload.content_type, image, payload.blob_pathname, scan_id

    raise HTTPException(status_code=415, detail="Upload an image using multipart form data or a private blob reference.")


def build_router(model_api) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["patients-and-scans"])

    @router.post("/patients", status_code=201)
    def create_patient(payload: PatientCreate):
        _require_database()
        if not payload.consent:
            raise HTTPException(status_code=400, detail="Consent is required before processing an image.")
        if payload.gender not in {"Female", "Male", "Other"}:
            raise HTTPException(status_code=422, detail="Gender must be Female, Male, or Other.")
        access_token = secrets.token_urlsafe(32)
        with SessionLocal() as db:
            patient = Patient(**payload.model_dump(), access_token_hash=_token_hash(access_token))
            db.add(patient)
            db.commit()
            db.refresh(patient)
            response = _patient_dict(patient)
            response["access_token"] = access_token  # returned once; only the hash is stored
            return response

    @router.get("/patients")
    def list_patients(x_admin_access_token: str | None = Header(default=None)):
        _require_database()
        admin_token = os.getenv("ADMIN_ACCESS_TOKEN", "")
        if not admin_token or not x_admin_access_token or not hmac.compare_digest(admin_token, x_admin_access_token):
            raise HTTPException(status_code=404, detail="Not found.")
        with SessionLocal() as db:
            patients = db.query(Patient).order_by(Patient.created_at.desc()).all()
            return {"patients": [_patient_dict(patient) for patient in patients]}

    @router.get("/patients/{patient_id}")
    def get_patient(patient_id: str, x_patient_access_token: str | None = Header(default=None)):
        _require_database()
        with SessionLocal() as db:
            patient = _get_patient(db, patient_id)
            if patient is None:
                raise HTTPException(status_code=404, detail="Patient record not found.")
            _authorize_patient(patient, x_patient_access_token)
            return _patient_dict(patient)

    @router.get("/patients/{patient_id}/scans")
    def get_patient_scans(patient_id: str, x_patient_access_token: str | None = Header(default=None)):
        _require_database()
        with SessionLocal() as db:
            patient = _get_patient(db, patient_id)
            if patient is None:
                raise HTTPException(status_code=404, detail="Patient record not found.")
            _authorize_patient(patient, x_patient_access_token)
            scans = db.query(Scan).filter(Scan.patient_id == patient.id).order_by(Scan.created_at.desc()).all()
            return {"patient": _patient_dict(patient), "scans": [_scan_dict(scan) for scan in scans]}

    @router.post("/patients/{patient_id}/upload-grant")
    def create_upload_grant(patient_id: str, payload: UploadGrantRequest,
                            x_patient_access_token: str | None = Header(default=None)):
        _require_database()
        if STORAGE_BACKEND != "blob":
            raise HTTPException(status_code=400, detail="Direct uploads are only enabled with private Blob storage.")
        if payload.kind not in {"image", "report"}:
            raise HTTPException(status_code=422, detail="Upload kind must be image or report.")
        valid_types = {"image/jpeg", "image/png"} if payload.kind == "image" else {"application/pdf"}
        if payload.content_type not in valid_types:
            raise HTTPException(status_code=415, detail="Unsupported upload content type.")
        try:
            scan_id = str(uuid.UUID(payload.scan_id))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid scan ID.") from exc
        with SessionLocal() as db:
            patient = _get_patient(db, patient_id)
            if patient is None:
                raise HTTPException(status_code=404, detail="Patient record not found.")
            _authorize_patient(patient, x_patient_access_token)
            if not patient.consent:
                raise HTTPException(status_code=400, detail="Patient consent is not recorded.")
            existing_scan = db.get(Scan, scan_id)
            if payload.kind == "image" and existing_scan is not None:
                raise HTTPException(status_code=409, detail="Scan ID already exists.")
            if payload.kind == "report" and (existing_scan is None or existing_scan.patient_id != patient.id):
                raise HTTPException(status_code=404, detail="Scan record not found.")
            if payload.kind == "image":
                ext = ".png" if payload.content_type == "image/png" else ".jpg"
                pathname = f"patients/{patient.id}/scans/{scan_id}/original{ext}"
                maximum_size = MAX_IMAGE_BYTES
            else:
                pathname = f"patients/{patient.id}/scans/{scan_id}/report.pdf"
                maximum_size = MAX_REPORT_BYTES
            payload_data = {
                "pathname": pathname,
                "content_type": payload.content_type,
                "kind": payload.kind,
                "maximum_size": maximum_size,
                "expires_at": int(datetime.now(timezone.utc).timestamp()) + 900,
            }
            return {"pathname": pathname, "upload_grant": _signed_payload(payload_data), "maximum_size": maximum_size}

    @router.post("/patients/{patient_id}/scans", status_code=201)
    async def create_scan(patient_id: str, request: Request,
                          x_patient_access_token: str | None = Header(default=None)):
        _require_database()
        with SessionLocal() as db:
            patient = _get_patient(db, patient_id)
            if patient is None:
                raise HTTPException(status_code=404, detail="Patient record not found.")
            _authorize_patient(patient, x_patient_access_token)
            if not patient.consent:
                raise HTTPException(status_code=400, detail="Patient consent is not recorded.")
            stored_patient_id = patient.id

        raw, content_type, image, uploaded_blob_path, requested_scan_id = await _request_image(
            request, model_api, stored_patient_id
        )
        scan_id = requested_scan_id or str(uuid.uuid4())
        sanitized_stream = BytesIO()
        if content_type == "image/png":
            image.save(sanitized_stream, format="PNG", optimize=True)
        else:
            image.save(sanitized_stream, format="JPEG", quality=90, optimize=True)
        sanitized_bytes = sanitized_stream.getvalue()
        if len(sanitized_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Sanitized image exceeds the 20 MB storage limit.")
        if uploaded_blob_path:
            ext = ".png" if content_type == "image/png" else ".jpg"
            expected_path = f"patients/{stored_patient_id}/scans/{scan_id}/original{ext}"
            if uploaded_blob_path != expected_path:
                raise HTTPException(status_code=403, detail="Uploaded image does not belong to this patient.")
            await _replace_private_blob(uploaded_blob_path, sanitized_bytes, content_type)
            original_path = f"blob://{uploaded_blob_path}"
        else:
            filename = f"{scan_id}{'.png' if content_type == 'image/png' else '.jpg'}"
            original_path = await _store_asset("originals", stored_patient_id, scan_id, filename, sanitized_bytes, content_type)

        result = model_api.analyze_bytes(raw, content_type)
        overlay_path = None
        heatmap_path = None
        gradcam = result.get("gradcam")
        if result.get("valid_cervical_image") and gradcam is not None:
            cam = np.clip(np.asarray(gradcam, dtype=np.float32), 0, 1)
            cam_u8 = np.uint8(cam * 255)
            cam_resized = cv2.resize(cam_u8, (image.width, image.height), interpolation=cv2.INTER_CUBIC)
            color = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
            original_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(original_bgr, 0.58, color, 0.42, 0)
            ok_overlay, overlay_encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
            ok_heatmap, heatmap_encoded = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok_overlay and ok_heatmap:
                overlay_path = await _store_asset(
                    "overlays", stored_patient_id, scan_id, f"{scan_id}.jpg", overlay_encoded.tobytes(), "image/jpeg"
                )
                heatmap_path = await _store_asset(
                    "heatmaps", stored_patient_id, scan_id, f"{scan_id}.jpg", heatmap_encoded.tobytes(), "image/jpeg"
                )

        with SessionLocal() as db:
            if db.get(Scan, scan_id) is not None:
                raise HTTPException(status_code=409, detail="Scan ID already exists.")
            scan = Scan(
                id=scan_id,
                patient_id=stored_patient_id,
                original_path=original_path,
                overlay_path=overlay_path,
                heatmap_path=heatmap_path,
                prediction_json=json.dumps(result),
                predicted_type=result.get("predicted_type"),
                confidence=result.get("confidence"),
                quality_json=json.dumps({"status": "not_assessed"}),
                metrics_json=json.dumps({
                    "status": "not_computed",
                    "reason": "No validated cervix/lesion segmentation mask is currently available.",
                }),
            )
            db.add(scan)
            db.commit()
            db.refresh(scan)
            return {"scan": _scan_dict(scan), "result": result}

    @router.get("/scans/{scan_id}")
    def get_scan(scan_id: str, x_patient_access_token: str | None = Header(default=None)):
        _require_database()
        with SessionLocal() as db:
            scan = _get_scan(db, scan_id)
            if scan is None:
                raise HTTPException(status_code=404, detail="Scan record not found.")
            patient = _get_patient(db, scan.patient_id)
            if patient is None:
                raise HTTPException(status_code=404, detail="Patient record not found.")
            _authorize_patient(patient, x_patient_access_token)
            return {"scan": _scan_dict(scan), "patient": _patient_dict(patient)}

    @router.get("/scans/{scan_id}/assets/{kind}")
    def get_scan_asset(scan_id: str, kind: str,
                       x_patient_access_token: str | None = Header(default=None)):
        _require_database()
        column_by_kind = {
            "original": "original_path",
            "overlay": "overlay_path",
            "heatmap": "heatmap_path",
            "report": "report_path",
        }
        if kind not in column_by_kind:
            raise HTTPException(status_code=404, detail="Asset not found.")
        with SessionLocal() as db:
            scan = _get_scan(db, scan_id)
            if scan is None:
                raise HTTPException(status_code=404, detail="Scan record not found.")
            patient = _get_patient(db, scan.patient_id)
            if patient is None:
                raise HTTPException(status_code=404, detail="Patient record not found.")
            _authorize_patient(patient, x_patient_access_token)
            path = getattr(scan, column_by_kind[kind])
            if not _path_exists(path):
                raise HTTPException(status_code=404, detail="Asset not found.")
            if path.startswith("blob://"):
                return {"blob_pathname": path.removeprefix("blob://"), "kind": kind}
            return {"local_path": path, "kind": kind}

    @router.post("/scans/{scan_id}/report", status_code=201)
    async def save_report(scan_id: str, request: Request,
                          x_patient_access_token: str | None = Header(default=None)):
        _require_database()
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        with SessionLocal() as db:
            scan = _get_scan(db, scan_id)
            if scan is None:
                raise HTTPException(status_code=404, detail="Scan record not found.")
            patient = _get_patient(db, scan.patient_id)
            if patient is None:
                raise HTTPException(status_code=404, detail="Patient record not found.")
            _authorize_patient(patient, x_patient_access_token)
            if media_type == "application/json" and STORAGE_BACKEND == "blob":
                try:
                    body = await request.json()
                    blob_pathname = str(body.get("blob_pathname", ""))
                except Exception as exc:
                    raise HTTPException(status_code=422, detail="A private report upload reference is required.") from exc
                expected_path = f"patients/{patient.id}/scans/{scan.id}/report.pdf"
                if blob_pathname != expected_path:
                    raise HTTPException(status_code=403, detail="Uploaded report does not belong to this scan.")
                raw, stored_type = await _read_blob(blob_pathname, max_bytes=MAX_REPORT_BYTES)
                if stored_type.split(";", 1)[0].strip().lower() != "application/pdf":
                    raise HTTPException(status_code=415, detail="Stored report is not a PDF.")
                path = f"blob://{blob_pathname}"
            elif media_type == "application/pdf" and STORAGE_BACKEND == "local":
                length = request.headers.get("content-length")
                if length and int(length) > MAX_REPORT_BYTES:
                    raise HTTPException(status_code=413, detail="PDF exceeds the 20 MB limit.")
                raw = await request.body()
                if len(raw) > MAX_REPORT_BYTES:
                    raise HTTPException(status_code=413, detail="PDF exceeds the 20 MB limit.")
                path = await _store_asset("reports", patient.id, scan.id, f"{scan.id}.pdf", raw, "application/pdf")
            else:
                raise HTTPException(status_code=415, detail="Report must be uploaded as a PDF.")
            if not raw.startswith(b"%PDF-"):
                raise HTTPException(status_code=400, detail="Report content is not a valid PDF.")
            scan.report_path = path
            db.commit()
            return {"scan_id": scan.id, "saved": True, "has_report": True}

    @router.get("/scans/{scan_id}/report")
    async def get_report(scan_id: str, x_patient_access_token: str | None = Header(default=None)):
        _require_database()
        with SessionLocal() as db:
            scan = _get_scan(db, scan_id)
            if scan is None:
                raise HTTPException(status_code=404, detail="Scan record not found.")
            patient = _get_patient(db, scan.patient_id)
            if patient is None:
                raise HTTPException(status_code=404, detail="Patient record not found.")
            _authorize_patient(patient, x_patient_access_token)
            path = scan.report_path
            if not path or not _path_exists(path):
                raise HTTPException(status_code=404, detail="Report not found.")
            if path.startswith("blob://"):
                raw, _ = await _load_blob(path)
            else:
                raw = Path(path).read_bytes()
        if len(raw) > 4 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Use the private asset URL endpoint for reports over 4 MB.")
        return StreamingResponse(
            iter([raw]), media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="scan-{scan_id}.pdf"', "Cache-Control": "private, no-store"},
        )

    @router.get("/facilities")
    def get_facilities(city: str | None = None, pincode: str | None = None,
                       lat: float | None = None, lng: float | None = None):
        path = BASE_DIR / "data" / "facilities.json"
        if not path.exists():
            return {"facilities": [], "note": "No facilities have been configured."}
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="facilities.json is not valid JSON.") from exc
        if city:
            rows = [row for row in rows if city.casefold() in str(row.get("city", "")).casefold()]
        if pincode:
            rows = [row for row in rows if str(row.get("pincode", "")).strip() == pincode.strip()]
        if lat is not None and lng is not None:
            import math
            def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
                radius = 6371.0
                dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
                a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
                return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            ranked = []
            for row in rows:
                item = dict(row)
                item["distance_km"] = (
                    round(haversine(lat, lng, float(item["lat"]), float(item["lng"])), 1)
                    if item.get("lat") is not None and item.get("lng") is not None else None
                )
                ranked.append(item)
            rows = sorted(ranked, key=lambda item: (item["distance_km"] is None, item["distance_km"] or 0))
        return {"facilities": rows, "note": "Please call ahead to confirm clinic timings and referral requirements."}

    return router
