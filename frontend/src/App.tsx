import { lazy, Suspense, useRef, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import "./App.css";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ExternalLink,
  FileText,
  ImagePlus,
  Info,
  LoaderCircle,
  MapPin,
  Phone,
  RotateCcw,
  ShieldCheck,
  Upload,
  UserRound,
  X,
} from "lucide-react";

const Cervix3DViewer = lazy(() => import("./Cervix3DViewer"));

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/+$/, "");
const MAX_FILE_SIZE = 20 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/png"];

type PatientForm = {
  first_name: string;
  last_name: string;
  age: string;
  gender: "Female";
  case_ref: string;
  notes: string;
  consent: boolean;
};

type PatientResponse = {
  patient_id?: string;
  id?: string;
  access_token?: string;
  patient?: {
    patient_id?: string;
    id?: string;
    access_token?: string;
  };
  detail?: string | { msg?: string }[];
};

type AnalysisResponse = {
  valid_cervical_image?: boolean;
  message?: string;
  predicted_type?: string;
  confidence?: number;
  probabilities?: Record<string, number>;
  interpretation?: string;
  suggested_next_step?: string;
  gradcam?: number[][];
  disclaimer?: string;
  detail?: string | { msg?: string }[];
};

type ScanResponse = {
  scan: {
    id: string;
    patient_id: string;
    predicted_type?: string;
    confidence?: number;
    prediction: AnalysisResponse;
    created_at?: string;
    has_original?: boolean;
    has_overlay?: boolean;
    has_heatmap?: boolean;
    has_report?: boolean;
  };
};

type PatientHistoryResponse = {
  patient: { id: string; name: string };
  scans: ScanResponse["scan"][];
};

type Facility = {
  id: string;
  name: string;
  type: string;
  address: string;
  city: string;
  state: string;
  pincode: string;
  phone: string;
  website: string;
  specialities: string[];
  lat: number | null;
  lng: number | null;
  verified: boolean;
  area?: string;
  locality?: string;
  distance_km?: number | null;
};

const initialForm: PatientForm = {
  first_name: "",
  last_name: "",
  age: "",
  gender: "Female",
  case_ref: "",
  notes: "",
  consent: false,
};

function getErrorMessage(data: PatientResponse | AnalysisResponse) {
  if (typeof data.detail === "string") return data.detail;

  if (Array.isArray(data.detail)) {
    return data.detail.map((item) => item.msg || "Invalid request").join("; ");
  }

  if ("message" in data && typeof data.message === "string") {
    return data.message;
  }

  return "Something went wrong. Please try again.";
}

export default function App() {
  const [form, setForm] = useState<PatientForm>(initialForm);
  const [patientId, setPatientId] = useState("");
  const [patientAccessToken, setPatientAccessToken] = useState("");
  const [step, setStep] = useState<"patient" | "upload" | "results">("patient");
  const [patientLoading, setPatientLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [patientError, setPatientError] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState("");
  const [aboutOpen, setAboutOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyPatientId, setHistoryPatientId] = useState("");
  const [historyAccessToken, setHistoryAccessToken] = useState("");
  const [historyPatient, setHistoryPatient] = useState<PatientHistoryResponse["patient"] | null>(null);
  const [historyScans, setHistoryScans] = useState<PatientHistoryResponse["scans"] | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [showAccessToken, setShowAccessToken] = useState(false);
  const [analysisResponse, setAnalysisResponse] =
    useState<AnalysisResponse | null>(null);

  const [savedScan, setSavedScan] = useState<ScanResponse["scan"] | null>(null);

  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.5);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [reportSaveStatus, setReportSaveStatus] = useState("");

  const [facilitiesOpen, setFacilitiesOpen] = useState(false);
  const [facilitiesLoading, setFacilitiesLoading] = useState(false);
  const [facilities, setFacilities] = useState<Facility[]>([]);
  const [facilitiesQuery, setFacilitiesQuery] = useState("");
  const [facilitiesNote, setFacilitiesNote] = useState("");
  const [facilitiesError, setFacilitiesError] = useState("");
  const [selectedFacility, setSelectedFacility] = useState<Facility | null>(null);
  const [facilitiesUsed, setFacilitiesUsed] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  function drawGradCam(
    canvas: HTMLCanvasElement,
    gradcam: number[][],
    opacity: number,
  ) {
    if (!gradcam.length || !gradcam[0]?.length) return;

    const rows = gradcam.length;
    const cols = gradcam[0].length;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = cols;
    canvas.height = rows;

    const values = gradcam.flat().filter(Number.isFinite);
    if (!values.length) return;

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const image = ctx.createImageData(cols, rows);

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const value = Math.max(
          0,
          Math.min(1, (gradcam[y][x] - min) / range),
        );

        const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * value - 3)));
        const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * value - 2)));
        const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * value - 1)));

        const index = (y * cols + x) * 4;
        image.data[index] = Math.round(r * 255);
        image.data[index + 1] = Math.round(g * 255);
        image.data[index + 2] = Math.round(b * 255);
        image.data[index + 3] = Math.round(value * opacity * 255);
      }
    }

    ctx.putImageData(image, 0, 0);
  }

  async function generateGradCamReportImage(
    imageSrc: string | null,
    gradcam: number[][] | undefined,
    opacity: number = 0.55,
  ): Promise<{ dataUrl: string; width: number; height: number } | null> {
    if (!gradcam || !gradcam.length || !gradcam[0]?.length) return null;

    const rows = gradcam.length;
    const cols = gradcam[0].length;

    // 1. Render heatmap onto offscreen canvas using drawGradCam
    const heatCanvas = document.createElement("canvas");
    drawGradCam(heatCanvas, gradcam, opacity);

    // 2. If original image preview exists, composite the heatmap over it
    if (imageSrc) {
      try {
        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
          const el = new Image();
          el.crossOrigin = "anonymous";
          el.onload = () => resolve(el);
          el.onerror = (e) => reject(e);
          el.src = imageSrc;
        });

        const targetWidth = img.naturalWidth || 600;
        const targetHeight = img.naturalHeight || 450;

        const compositeCanvas = document.createElement("canvas");
        compositeCanvas.width = targetWidth;
        compositeCanvas.height = targetHeight;

        const ctx = compositeCanvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
          ctx.imageSmoothingEnabled = true;
          ctx.imageSmoothingQuality = "high";
          ctx.drawImage(heatCanvas, 0, 0, targetWidth, targetHeight);

          return {
            dataUrl: compositeCanvas.toDataURL("image/jpeg", 0.92),
            width: targetWidth,
            height: targetHeight,
          };
        }
      } catch (err) {
        console.warn("Could not composite with original image, falling back to standalone heatmap:", err);
      }
    }

    // 3. Fallback: Standalone scaled-up heatmap with white background
    const standaloneCanvas = document.createElement("canvas");
    const targetWidth = 400;
    const targetHeight = Math.round((400 * rows) / cols);
    standaloneCanvas.width = targetWidth;
    standaloneCanvas.height = targetHeight;
    const sCtx = standaloneCanvas.getContext("2d");
    if (sCtx) {
      sCtx.fillStyle = "#ffffff";
      sCtx.fillRect(0, 0, targetWidth, targetHeight);
      sCtx.imageSmoothingEnabled = true;
      sCtx.imageSmoothingQuality = "high";
      sCtx.drawImage(heatCanvas, 0, 0, targetWidth, targetHeight);
      return {
        dataUrl: standaloneCanvas.toDataURL("image/png"),
        width: targetWidth,
        height: targetHeight,
      };
    }

    return {
      dataUrl: heatCanvas.toDataURL("image/png"),
      width: cols,
      height: rows,
    };
  }

  const age = Number(form.age);
  const formIsValid =
    form.first_name.trim().length > 0 &&
    form.last_name.trim().length > 0 &&
    form.age.trim().length > 0 &&
    Number.isInteger(age) &&
    age >= 1 &&
    age <= 120 &&
    form.consent;

  async function fetchFacilities(
    query?: string,
    coords?: { lat: number; lng: number },
  ) {
    setFacilitiesLoading(true);
    setFacilitiesError("");
    try {
      let url = `${API_BASE}/api/facilities`;
      const params = new URLSearchParams();
      if (query && query.trim()) {
        const q = query.trim();
        if (/^\d+$/.test(q)) {
          params.append("pincode", q);
        } else {
          params.append("city", q);
        }
      }
      if (coords) {
        params.append("lat", String(coords.lat));
        params.append("lng", String(coords.lng));
      }
      const qs = params.toString();
      if (qs) url += `?${qs}`;

      const res = await fetch(url);
      if (!res.ok) throw new Error("Could not load care centres.");
      const data = await res.json();
      setFacilities(data.facilities || []);
      setFacilitiesNote(data.note || "");
    } catch (err) {
      setFacilitiesError(
        err instanceof Error ? err.message : "Failed to load facilities",
      );
    } finally {
      setFacilitiesLoading(false);
    }
  }

  function handleUseLocation() {
    if (!navigator.geolocation) {
      setFacilitiesError("Geolocation is not supported by your browser.");
      return;
    }
    setFacilitiesLoading(true);
    setFacilitiesError("");
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        fetchFacilities(facilitiesQuery, {
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
        });
      },
      () => {
        setFacilitiesError(
          "Location access was denied or timed out. You can still search by City or PIN code.",
        );
        setFacilitiesLoading(false);
      },
      { timeout: 8000 },
    );
  }

  function updateField<K extends keyof PatientForm>(
    key: K,
    value: PatientForm[K],
  ) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function handlePatientSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPatientError("");

    if (!formIsValid) {
      setPatientError(
        "Enter valid patient details and confirm consent before continuing.",
      );
      return;
    }

    setPatientLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/patients`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          first_name: form.first_name.trim(),
          last_name: form.last_name.trim(),
          age,
          gender: "Female",
          case_ref: form.case_ref.trim() || null,
          notes: form.notes.trim() || null,
          consent: form.consent,
        }),
      });

      const data: PatientResponse = await response.json();

      if (!response.ok) {
        throw new Error(getErrorMessage(data));
      }

      const id =
        data.patient_id ??
        data.id ??
        data.patient?.patient_id ??
        data.patient?.id;

      if (!id) {
        throw new Error(
          "The backend accepted the request but did not return a patient ID.",
        );
      }

      const accessToken = data.access_token ?? data.patient?.access_token;
      if (!accessToken) {
        throw new Error("The backend did not return the patient access token. The record cannot be used safely.");
      }

      setPatientId(String(id));
      setPatientAccessToken(accessToken);
      setShowAccessToken(false);
      setStep("upload");
      setPatientError("");
    } catch (error) {
      setPatientError(
        error instanceof Error
          ? error.message
          : "Unable to create the patient record. Check that the backend is running.",
      );
    } finally {
      setPatientLoading(false);
    }
  }

  async function handleHistorySubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setHistoryLoading(true);
    setHistoryError("");
    setHistoryPatient(null);
    setHistoryScans(null);
    try {
      const response = await fetch(
        `${API_BASE}/api/patients/${encodeURIComponent(historyPatientId.trim())}/scans`,
        { headers: { "X-Patient-Access-Token": historyAccessToken.trim() } },
      );
      const data = await response.json() as PatientHistoryResponse & {
        detail?: string | { msg?: string }[];
      };
      if (!response.ok) {
        throw new Error(getErrorMessage(data));
      }
      setHistoryPatient(data.patient);
      setHistoryScans(data.scans);
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "Could not load scan history.");
    } finally {
      setHistoryLoading(false);
    }
  }

  async function downloadHistoryReport(scanId: string) {
    setHistoryError("");
    try {
      if (import.meta.env.PROD) {
        const response = await fetch(`${API_BASE}/api/private-asset`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Patient-Access-Token": historyAccessToken.trim(),
          },
          body: JSON.stringify({ scan_id: scanId, kind: "report" }),
        });
        const data = await response.json() as { url?: string; error?: string };
        if (!response.ok || !data.url) throw new Error(data.error || "Report is unavailable.");
        window.open(data.url, "_blank", "noopener,noreferrer");
      } else {
        const response = await fetch(`${API_BASE}/api/scans/${encodeURIComponent(scanId)}/report`, {
          headers: { "X-Patient-Access-Token": historyAccessToken.trim() },
        });
        if (!response.ok) {
          const data = await response.json() as { detail?: string };
          throw new Error(data.detail || "Report is unavailable.");
        }
        const url = URL.createObjectURL(await response.blob());
        const link = document.createElement("a");
        link.href = url;
        link.download = `scan-${scanId}.pdf`;
        link.click();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "Could not retrieve the report.");
    }
  }

  function clearSelectedImage() {
    setImageFile(null);
    setImagePreview("");
    setUploadError("");
    setAnalysisResponse(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function handleImageSelection(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];

    setUploadError("");
    setAnalysisResponse(null);

    if (!file) return;

    if (!ALLOWED_TYPES.includes(file.type)) {
      setUploadError("Choose a JPG, JPEG, or PNG image.");
      event.target.value = "";
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setUploadError("The image must be 20 MB or smaller.");
      event.target.value = "";
      return;
    }

    if (imagePreview) URL.revokeObjectURL(imagePreview);

    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
  }

  async function handleAnalyzeImage() {
    if (!imageFile) {
      setUploadError("Select an image before continuing.");
      return;
    }

    setUploadError("");
    setAnalysisResponse(null);
    setUploadLoading(true);

    try {
      if (!patientId) {
        throw new Error("Patient ID is missing. Please create the patient record again.");
      }

      if (!patientAccessToken) {
        throw new Error("Patient access token is missing. Please create the patient record again.");
      }

      const scanEndpoint = `${API_BASE}/api/patients/${encodeURIComponent(patientId)}/scans`;
      let response: Response;
      if (import.meta.env.PROD) {
        const scanId = crypto.randomUUID();
        const grantResponse = await fetch(
          `${API_BASE}/api/patients/${encodeURIComponent(patientId)}/upload-grant`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-Patient-Access-Token": patientAccessToken,
            },
            body: JSON.stringify({ scan_id: scanId, content_type: imageFile.type, kind: "image" }),
          },
        );
        const grant = await grantResponse.json();
        if (!grantResponse.ok) throw new Error(getErrorMessage(grant));

        const { upload } = await import("@vercel/blob/client");
        const uploaded = await upload(grant.pathname, imageFile, {
          access: "private",
          contentType: imageFile.type,
          handleUploadUrl: `${API_BASE}/api/blob-upload`,
          clientPayload: grant.upload_grant,
          multipart: true,
        });
        if (uploaded.pathname !== grant.pathname) {
          throw new Error("The private upload returned an unexpected image path.");
        }
        response = await fetch(scanEndpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Patient-Access-Token": patientAccessToken,
          },
          body: JSON.stringify({
            scan_id: scanId,
            blob_pathname: uploaded.pathname,
            content_type: imageFile.type,
          }),
        });
      } else {
        const body = new FormData();
        body.append("file", imageFile);
        response = await fetch(scanEndpoint, {
          method: "POST",
          headers: { "X-Patient-Access-Token": patientAccessToken },
          body,
        });
      }

      const data = (await response.json()) as ScanResponse | AnalysisResponse;

      if (!response.ok) {
        throw new Error(
          getErrorMessage(data as AnalysisResponse),
        );
      }

      if (!("scan" in data) || !data.scan?.prediction) {
        throw new Error("The backend response did not contain the saved scan results.");
      }

      setSavedScan(data.scan);
      setAnalysisResponse(data.scan.prediction);

      if (data.scan.prediction.valid_cervical_image === false) {
        setUploadError(
          data.scan.prediction.message ||
          "The image was not accepted by the screening validator. Please check the image and try again.",
        );
        return;
      }

      setStep("results");


    } catch (error) {
      setUploadError(
        error instanceof Error
          ? error.message
          : "Unable to analyze the image. Check that the backend is running.",
      );
    } finally {
      setUploadLoading(false);
    }
  }

  function startAnotherPatient() {
    if (imagePreview) URL.revokeObjectURL(imagePreview);

    setForm(initialForm);
    setPatientId("");
    setPatientAccessToken("");
    setStep("patient");
    setImageFile(null);
    setImagePreview("");
    setPatientError("");
    setUploadError("");
    setAnalysisResponse(null);
    setSavedScan(null);
    setReportSaveStatus("");
  }

  return (
    <main className="page-shell">
      <section className={`screening-card ${step === "upload" ? "upload-card-layout" : ""}`}>
        <aside className="welcome-panel">
          <div className="brand">
            <div className="brand-icon">
              <Activity size={32} strokeWidth={2.2} />
            </div>
            <div className="brand-name">
              <strong>Early Cervical</strong>
              <span>Risk Detection</span>
            </div>
          </div>

          <div className="research-pill">
            <ShieldCheck size={18} />
            <span>Research &amp; education</span>
          </div>

          <div className="welcome-copy">
            <h1>Early detection saves lives.</h1>
            <p>
              AI-assisted colposcopy analysis with visual explanations.
            </p>
          </div>

          <div className="welcome-bottom-btns">
            <button
              type="button"
              className="pill-nav-btn"
              onClick={() => setAboutOpen(true)}
            >
              <div className="pill-btn-content">
                <BookOpen size={19} />
                <span>About the project</span>
              </div>
              <ArrowRight size={17} />
            </button>

            <button
              type="button"
              className="pill-nav-btn"
              onClick={() => {
                setFacilitiesOpen(true);
                setFacilitiesUsed(true);
                if (facilities.length === 0) fetchFacilities();
              }}
            >
              <div className="pill-btn-content">
                <MapPin size={19} />
                <span>Nearby care centres</span>
              </div>
              <ArrowRight size={17} />
            </button>
          </div>

          <div className="circle-one" />
          <div className="circle-two" />
          <div className="circle-three" />
        </aside>

        <section className="form-panel">
          {step === "patient" ? (
            <>
              <p className="eyebrow">PATIENT DETAILS</p>
              <h2 className="form-title">Start a screening</h2>
              <p className="form-intro">
                Enter the patient details to begin. You can upload a
                colposcopy image in the next step.
              </p>

              <form
                className="patient-form"
                onSubmit={handlePatientSubmit}
                noValidate
              >
                <div className="form-grid-two-col">
                  <div className="form-field">
                    <label htmlFor="first_name">
                      <span>First name <span className="req">*</span></span>
                    </label>
                    <input
                      id="first_name"
                      name="first_name"
                      autoComplete="given-name"
                      value={form.first_name}
                      onChange={(e) =>
                        updateField("first_name", e.target.value)
                      }
                      placeholder="Enter first name"
                      required
                    />
                  </div>

                  <div className="form-field">
                    <label htmlFor="last_name">
                      <span>Last name <span className="req">*</span></span>
                    </label>
                    <input
                      id="last_name"
                      name="last_name"
                      autoComplete="family-name"
                      value={form.last_name}
                      onChange={(e) =>
                        updateField("last_name", e.target.value)
                      }
                      placeholder="Enter last name"
                      required
                    />
                  </div>

                  <div className="form-field">
                    <label htmlFor="age">
                      <span>Age <span className="req">*</span></span>
                    </label>
                    <input
                      id="age"
                      name="age"
                      type="number"
                      min="1"
                      max="120"
                      step="1"
                      value={form.age}
                      onChange={(e) => updateField("age", e.target.value)}
                      placeholder="Enter age"
                      required
                    />
                  </div>

                  <div className="form-field">
                    <label htmlFor="gender">
                      <span>Gender <span className="req">*</span></span>
                    </label>
                    <div className="select-wrapper">
                      <select
                        id="gender"
                        name="gender"
                        value={form.gender}
                        onChange={(e) => updateField("gender", e.target.value as "Female")}
                      >
                        <option value="Female">Female</option>
                      </select>
                      <ChevronDown className="select-chevron" size={18} />
                    </div>
                  </div>

                  <div className="form-field">
                    <label htmlFor="case_ref">
                      <span>Case reference <span className="opt">(optional)</span></span>
                    </label>
                    <input
                      id="case_ref"
                      name="case_ref"
                      value={form.case_ref}
                      onChange={(e) =>
                        updateField("case_ref", e.target.value)
                      }
                      placeholder="e.g. CR-2024-001"
                    />
                  </div>

                  <div className="form-field">
                    <label htmlFor="notes">
                      <span>Clinical notes <span className="opt">(optional)</span></span>
                    </label>
                    <textarea
                      id="notes"
                      name="notes"
                      rows={1}
                      value={form.notes}
                      onChange={(e) => updateField("notes", e.target.value)}
                      placeholder="Brief notes about the case..."
                    />
                  </div>
                </div>

                <label className="custom-consent-row">
                  <input
                    type="checkbox"
                    checked={form.consent}
                    onChange={(e) => updateField("consent", e.target.checked)}
                    required
                  />
                  <span>
                    I confirm that I have obtained informed consent from the patient for using this data for research and education purposes. <span className="req">*</span>
                  </span>
                </label>

                {patientError && (
                  <div className="error-message" role="alert" style={{ marginBottom: "16px" }}>
                    {patientError}
                  </div>
                )}

                <button
                  className="primary-button"
                  type="submit"
                  disabled={!formIsValid || patientLoading}
                >
                  {patientLoading ? (
                    <>
                      <LoaderCircle className="spin" size={19} />
                      Creating patient record...
                    </>
                  ) : (
                    <>
                      Continue to image upload
                      <ArrowRight size={20} />
                    </>
                  )}
                </button>
              </form>

              <div className="screen-one-footer">
                <div className="footer-links-row">
                  <button
                    type="button"
                    className="footer-link-btn"
                    onClick={() => {
                      setHistoryPatientId(patientId);
                      setHistoryAccessToken(patientAccessToken);
                      setHistoryPatient(null);
                      setHistoryScans(null);
                      setHistoryError("");
                      setHistoryOpen(true);
                    }}
                  >
                    <FileText size={16} />
                    <span>View previous cases</span>
                  </button>
                  <button
                    type="button"
                    className="footer-link-btn"
                    onClick={() => setAboutOpen(true)}
                  >
                    <Info size={16} />
                    <span>Learn about screening</span>
                  </button>
                </div>
                <p className="footer-disclaimer-text">
                  For research and education only. This system does not provide a clinical diagnosis.
                </p>
              </div>
            </>
          ) : step === "upload" ? (
            <div className="upload-screen-container">
              <div className="stepper-header">
                <span className="eyebrow" style={{ margin: 0 }}>STEP 2 OF 3</span>
                <div className="stepper-bar" role="progressbar" aria-valuenow={2} aria-valuemin={1} aria-valuemax={3}>
                  <div className="stepper-seg filled" />
                  <div className="stepper-seg filled" />
                  <div className="stepper-seg" />
                </div>
              </div>

              <h2 className="form-title upload-title">Upload a colposcopy image</h2>
              <p className="form-intro upload-intro">
                Select a clear JPG or PNG image for the screening model. Review the preview before starting analysis.
              </p>

              <div className="patient-record-card">
                <div className="patient-card-left">
                  <CheckCircle2 size={24} className="check-green" />
                  <div className="patient-card-text">
                    <strong>Patient record created</strong>
                    <span className="patient-id-mono">Patient ID: {patientId}</span>
                    <span className="patient-access-code">
                      Access code: {showAccessToken ? patientAccessToken : "hidden"}
                    </span>
                  </div>
                </div>
                <div className="patient-card-actions">
                  <button
                    type="button"
                    className="patient-details-btn"
                    onClick={() => setShowAccessToken((current) => !current)}
                  >
                    {showAccessToken ? "Hide code" : "Show code"}
                  </button>
                  <button
                    type="button"
                    className="patient-details-btn"
                    onClick={() => setStep("patient")}
                  >
                    Details
                  </button>
                </div>
              </div>

              <div
                className={`upload-dropzone-styled ${imagePreview ? "has-preview" : ""}`}
                onClick={() => {
                  if (!imagePreview && fileInputRef.current) {
                    fileInputRef.current.click();
                  }
                }}
                onDragOver={(event) => {
                  event.preventDefault();
                  event.currentTarget.classList.add("is-dragover");
                }}
                onDragLeave={(event) => {
                  event.currentTarget.classList.remove("is-dragover");
                }}
                onDrop={(event) => {
                  event.preventDefault();
                  event.currentTarget.classList.remove("is-dragover");
                  const file = event.dataTransfer.files?.[0];
                  if (!file) return;

                  if (fileInputRef.current) {
                    const transfer = new DataTransfer();
                    transfer.items.add(file);
                    fileInputRef.current.files = transfer.files;
                    fileInputRef.current.dispatchEvent(
                      new Event("change", { bubbles: true })
                    );
                  }
                }}
              >
                {imagePreview ? (
                  <div className="dropzone-preview-wrap" onClick={(e) => e.stopPropagation()}>
                    <img
                      src={imagePreview}
                      alt="Selected colposcopy preview"
                      className="dropzone-preview-img"
                    />
                    <div className="preview-meta-row">
                      <div className="preview-meta-text">
                        <strong>{imageFile?.name}</strong>
                        <span>
                          {imageFile
                            ? `${(imageFile.size / (1024 * 1024)).toFixed(2)} MB`
                            : ""}
                        </span>
                      </div>
                      <button
                        type="button"
                        className="change-image-btn"
                        onClick={clearSelectedImage}
                        aria-label="Remove selected image"
                        title="Remove image"
                      >
                        <X size={15} />
                        <span>Remove</span>
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="dropzone-empty-state">
                    <div className="dropzone-icon-circle">
                      <ImagePlus size={28} />
                    </div>
                    <h3>Select your image</h3>
                    <p>Drag and drop an image here, or browse your files.</p>
                    <button
                      type="button"
                      className="choose-image-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        fileInputRef.current?.click();
                      }}
                    >
                      <Upload size={16} />
                      Choose image
                    </button>
                    <span className="dropzone-req-caption">
                      JPG or PNG · Maximum 20 MB
                    </span>
                  </div>
                )}

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,.jpg,.jpeg,.png"
                  className="visually-hidden-input"
                  onChange={handleImageSelection}
                />
              </div>

              <div className="upload-privacy-note">
                <ShieldCheck size={20} className="shield-icon" />
                <p>
                  Use an image you are authorised to process. Avoid including identifying information in the image where possible.
                </p>
              </div>

              {uploadError && (
                <div className="error-message" role="alert" style={{ marginBottom: "16px" }}>
                  {uploadError}
                </div>
              )}

              {analysisResponse && analysisResponse.valid_cervical_image !== false && (
                <div className="upload-success" role="status" style={{ marginBottom: "16px" }}>
                  <CheckCircle2 size={22} />
                  <div>
                    <strong>Image accepted by the analysis endpoint</strong>
                    <span>
                      {analysisResponse.predicted_type
                        ? `Predicted transformation-zone type: ${analysisResponse.predicted_type}`
                        : "The backend returned a response."}
                    </span>
                  </div>
                </div>
              )}

              <button
                className="primary-button"
                type="button"
                onClick={handleAnalyzeImage}
                disabled={!imageFile || uploadLoading}
              >
                {uploadLoading ? (
                  <>
                    <LoaderCircle className="spin" size={19} />
                    Analyzing image...
                  </>
                ) : (
                  <>
                    Analyze image
                    <ArrowRight size={20} />
                  </>
                )}
              </button>

              <div className="upload-actions-row">
                <button
                  type="button"
                  className="secondary-ghost-btn"
                  onClick={() => setStep("patient")}
                >
                  <ArrowLeft size={16} />
                  <span>Back to patient details</span>
                </button>

                <button
                  type="button"
                  className="secondary-ghost-btn"
                  onClick={startAnotherPatient}
                >
                  <RotateCcw size={16} />
                  <span>New patient</span>
                </button>
              </div>

              <div className="screen-two-footer">
                <p className="footer-disclaimer-text">
                  Research and education only. Not a clinical diagnosis.
                </p>
              </div>
            </div>
          ) : (
            <>

              <p className="eyebrow">STEP 3 OF 3</p>
              <h2 className="form-title">Analysis results</h2>
              <p className="form-intro">
                Review the model output and its visual explanation.
              </p>

              <div className="patient-reference">
                <div className="reference-icon">
                  <CheckCircle2 size={21} />
                </div>
                <div className="reference-copy">
                  <strong>{form.first_name} {form.last_name}</strong>
                  <span>Patient ID: {patientId}</span>
                  <span>Scan ID: {savedScan?.id ?? "Not available"}</span>
                </div>
              </div>

              {analysisResponse?.valid_cervical_image === false ? (
                <div className="error-message" role="alert">
                  {analysisResponse.message ||
                    "This image was not accepted as a valid cervical image. No transformation-zone result should be interpreted."}
                </div>
              ) : (
                <>
                  <div className="workflow-note">
                    <Activity size={22} />
                    <div>
                      <strong>
                        Predicted type:{" "}
                        {savedScan?.predicted_type ??
                          analysisResponse?.predicted_type ??
                          "Not available"}
                      </strong>
                      <span>
                        Model confidence:{" "}
                        {typeof (savedScan?.confidence ??
                          analysisResponse?.confidence) === "number"
                          ? `${(
                            (savedScan?.confidence ??
                              analysisResponse?.confidence ??
                              0) * 100
                          ).toFixed(1)}%`
                          : "Not available"}
                      </span>
                    </div>
                  </div>

                  {typeof (savedScan?.confidence ??
                    analysisResponse?.confidence) === "number" &&
                    (savedScan?.confidence ??
                      analysisResponse?.confidence ??
                      0) < 0.60 && (
                      <div
                        className="low-confidence-warning"
                        style={{
                          display: "flex",
                          alignItems: "flex-start",
                          gap: "10px",
                          padding: "12px 16px",
                          margin: "12px 0",
                          borderRadius: "10px",
                          background: "#fffbeb",
                          border: "1px solid #fcd34d",
                          borderLeft: "5px solid #d97706",
                          color: "#92400e",
                        }}
                      >
                        <AlertTriangle
                          size={20}
                          style={{ flexShrink: 0, marginTop: "2px" }}
                        />
                        <div style={{ fontSize: "14px", lineHeight: "1.5" }}>
                          <strong
                            style={{
                              display: "block",
                              color: "#b45309",
                              marginBottom: "3px",
                            }}
                          >
                            Low Confidence Notice (&lt; 60%)
                          </strong>
                          The model's confidence in this classification is below
                          60%. This may indicate a borderline case, suboptimal
                          image illumination or contrast, or anatomy requiring
                          close manual clinical inspection.
                        </div>
                      </div>
                    )}

                  <div className="results-probabilities">
                    <h3>Class probabilities</h3>
                    {Object.entries(analysisResponse?.probabilities ?? {}).map(
                      ([label, probability]) => (
                        <div className="probability-row" key={label}>
                          <span>
                            {label}
                            {label ===
                              (savedScan?.predicted_type ??
                                analysisResponse?.predicted_type)
                              ? " · Predicted"
                              : ""}
                          </span>
                          <strong>{(probability * 100).toFixed(1)}%</strong>
                          <div className="probability-track">
                            <div
                              className="probability-fill"
                              style={{
                                width: `${Math.max(
                                  0,
                                  Math.min(100, probability * 100),
                                )}%`,
                              }}
                            />
                          </div>
                        </div>
                      ),
                    )}
                  </div>


                  {imagePreview && (
                    <div className="results-image-card">
                      <h3>Colposcopy image and Grad-CAM explanation</h3>
                      <div className="gradcam-preview">
                        <img
                          src={imagePreview}
                          alt="Original colposcopy image"
                          className="gradcam-original"
                        />

                        {showHeatmap && analysisResponse?.gradcam && (
                          <canvas
                            ref={(canvas) => {
                              if (canvas && analysisResponse.gradcam) {
                                drawGradCam(
                                  canvas,
                                  analysisResponse.gradcam,
                                  heatmapOpacity,
                                );
                              }
                            }}
                            className="gradcam-canvas"
                            data-gradcam-report="true"
                            aria-label="Grad-CAM heatmap overlay"
                          />
                        )}
                      </div>
                      <div className="heatmap-toggle-row">
                        <button
                          type="button"
                          className={showHeatmap ? "secondary-button" : "primary-button"}
                          onClick={() => setShowHeatmap(true)}
                        >
                          Heatmap overlay
                        </button>
                        <button
                          type="button"
                          className={!showHeatmap ? "secondary-button" : "primary-button"}
                          onClick={() => setShowHeatmap(false)}
                        >
                          Original image
                        </button>
                      </div>

                      {showHeatmap && (
                        <label className="heatmap-opacity-control">
                          <span>Heatmap opacity: {Math.round(heatmapOpacity * 100)}%</span>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={heatmapOpacity}
                            onChange={(event) =>
                              setHeatmapOpacity(Number(event.target.value))
                            }
                          />
                        </label>
                      )}

                      {showHeatmap && (
                        <div
                          className="gradcam-legend-bar"
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            gap: "8px",
                            margin: "8px 0 4px",
                            fontSize: "12px",
                            color: "#4b5f63",
                          }}
                        >
                          <span style={{ fontSize: "11px" }}>Lower influence</span>
                          <div
                            style={{
                              width: "120px",
                              height: "7px",
                              borderRadius: "4px",
                              background:
                                "linear-gradient(to right, #1d4ed8, #06b6d4, #22c55e, #eab308, #ef4444)",
                            }}
                          />
                          <span style={{ fontSize: "11px" }}>Higher influence</span>
                        </div>
                      )}

                      <p className="heatmap-caption">
                        Highlighted regions show areas that most influenced the model's
                        prediction. Grad-CAM is a model-influence visualization, not lesion segmentation.
                      </p>
                    </div>
                  )}

                  <Suspense
                    fallback={
                      <div className="results-image-card">
                        <h3>Interactive 3D anatomy</h3>
                        <p>Loading the educational anatomy model…</p>
                      </div>
                    }
                  >
                    <Cervix3DViewer />
                  </Suspense>

                  <div className="workflow-note">
                    <div>
                      <strong>Lesion assessment</strong>
                      <span>
                        Dedicated lesion detection is not available in this prototype. A clean-looking image or the absence of highlighted regions does not confirm that the cervix is lesion-free.
                      </span>
                    </div>
                  </div>

                  <div className="workflow-note clinical-analysis">
                    <div>
                      <strong>Clinical Analysis</strong>
                      <span>
                        {analysisResponse?.valid_cervical_image
                          ? analysisResponse.interpretation ??
                          "Transformation zone classification is available, but interpretation is unavailable."
                          : "Clinical interpretation is unavailable because this image was not accepted as a valid cervical image."}
                      </span>
                    </div>
                  </div>


                  <div className="workflow-note">
                    <div>
                      <strong>Suggested next step</strong>
                      <span>
                        {analysisResponse?.suggested_next_step ??
                          "No next step was returned."}
                      </span>
                      <button
                        type="button"
                        className="secondary-button"
                        style={{
                          marginTop: "12px",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "6px",
                          padding: "8px 14px",
                          fontSize: "13px",
                        }}
                        onClick={() => {
                          setFacilitiesOpen(true);
                          setFacilitiesUsed(true);
                          if (facilities.length === 0) fetchFacilities();
                        }}
                      >
                        <MapPin size={16} />
                        Find nearby referral centres
                      </button>
                    </div>
                  </div>
                </>
              )}

              <div className="panel-footer">
                <span>
                  <ShieldCheck size={16} />
                  Research and education only. Not a clinical diagnosis.
                </span>
              </div>

              <button
                type="button"
                className="secondary-button"
                disabled={pdfLoading}
                onClick={async () => {
                  if (!analysisResponse) return;
                  setPdfLoading(true);

                  try {
                    const { default: JsPDF } = await import("jspdf");
                    const pdf = new JsPDF({
                      orientation: "portrait",
                      unit: "mm",
                      format: "a4",
                    });

                    const pageWidth = 210;
                    const margin = 12;
                    const contentWidth = pageWidth - margin * 2; // 186mm

                    // Top title header
                    const drawHeader = () => {
                      pdf.setFillColor(238, 248, 248);
                      pdf.setDrawColor(204, 232, 231);
                      pdf.setLineWidth(0.3);
                      pdf.roundedRect(margin, 10, contentWidth, 14.5, 3, 3, "FD");

                      pdf.setFont("helvetica", "bold");
                      pdf.setFontSize(12.5);
                      pdf.setTextColor(10, 79, 87);
                      pdf.text("MEDICAL REPORT", pageWidth / 2, 16.8, { align: "center" });

                      pdf.setFont("helvetica", "bold");
                      pdf.setFontSize(6.5);
                      pdf.setTextColor(14, 124, 134);
                      pdf.text(
                        "CERVICAL IMAGE ANALYSIS - EXPLAINABLE COLPOSCOPY",
                        pageWidth / 2,
                        21.2,
                        { align: "center" }
                      );
                    };

                    // Bottom footer on every page
                    const drawFooter = () => {
                      pdf.setDrawColor(229, 231, 235);
                      pdf.setLineWidth(0.2);
                      pdf.line(margin, 287, pageWidth - margin, 287);

                      pdf.setFont("helvetica", "normal");
                      pdf.setFontSize(7.5);
                      pdf.setTextColor(130, 135, 150);
                      pdf.text(
                        "Research and education only. Not a clinical diagnosis.",
                        margin,
                        291.5
                      );
                      pdf.text(
                        `Page ${pdf.getNumberOfPages()}`,
                        pageWidth - margin,
                        291.5,
                        { align: "right" }
                      );
                    };

                    // Helper: Draw Section Card with Numbered Badge, Title, Sub-label, White Box with Teal Accent Bar
                    const drawSectionCard = (
                      startY: number,
                      sectionNum: string,
                      title: string,
                      subLabel: string,
                      boxHeight: number
                    ) => {
                      const cardHeight = 16.5 + boxHeight + 3.5;

                      // Auto page break check
                      if (startY + cardHeight > 284) {
                        drawFooter();
                        pdf.addPage();
                        drawHeader();
                        startY = 28;
                      }

                      // 1. Outer rounded card (soft teal-tinted background)
                      pdf.setFillColor(244, 249, 249);
                      pdf.setDrawColor(213, 236, 235);
                      pdf.setLineWidth(0.3);
                      pdf.roundedRect(margin, startY, contentWidth, cardHeight, 3.5, 3.5, "FD");

                      // 2. Numbered Teal Badge
                      const badgeX = margin + 4;
                      const badgeY = startY + 3.5;
                      pdf.setFillColor(14, 124, 134);
                      pdf.roundedRect(badgeX, badgeY, 5.2, 5.2, 1.2, 1.2, "F");

                      pdf.setFont("helvetica", "bold");
                      pdf.setFontSize(7.8);
                      pdf.setTextColor(255, 255, 255);
                      pdf.text(sectionNum, badgeX + 2.6, badgeY + 3.8, { align: "center" });

                      // 3. Section Title
                      pdf.setFont("helvetica", "bold");
                      pdf.setFontSize(10);
                      pdf.setTextColor(15, 42, 46);
                      pdf.text(title, badgeX + 7.8, badgeY + 4);

                      // 4. Sub-label
                      const subY = startY + 12.2;
                      pdf.setFont("helvetica", "bold");
                      pdf.setFontSize(6.8);
                      pdf.setTextColor(14, 124, 134);
                      pdf.text(subLabel, margin + 4, subY);

                      // 5. White Content Box
                      const boxX = margin + 4;
                      const boxY = startY + 14.5;
                      const boxW = contentWidth - 8;

                      pdf.setFillColor(255, 255, 255);
                      pdf.setDrawColor(229, 231, 235);
                      pdf.setLineWidth(0.25);
                      pdf.roundedRect(boxX, boxY, boxW, boxHeight, 2, 2, "FD");

                      // 6. Teal Left Accent Bar
                      pdf.setFillColor(14, 124, 134);
                      pdf.roundedRect(boxX, boxY, 1.6, boxHeight, 1, 1, "F");

                      return {
                        innerX: boxX + 4.5,
                        innerY: boxY + 3,
                        innerW: boxW - 7,
                        nextY: startY + cardHeight + 3.2,
                      };
                    };

                    // Draw initial page header
                    drawHeader();
                    let currentY = 27.5;

                    // ==========================================
                    // SECTION 1: Patient Information
                    // ==========================================
                    const s1BoxH = 21.5;
                    const s1 = drawSectionCard(
                      currentY,
                      "1",
                      "Patient Information",
                      "1  PATIENT DETAILS",
                      s1BoxH
                    );

                    const patientName = `${form.first_name} ${form.last_name}`.trim() || "Unavailable";
                    const patientAge = form.age ? `${form.age} yrs` : "Unavailable";
                    const patientGender = form.gender || "Unavailable";
                    const caseRef = form.case_ref || "Unavailable";
                    const patientIdVal = savedScan?.patient_id || patientId || "Unavailable";
                    const reportDate = new Date().toLocaleDateString();
                    const reportTime = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

                    const col1X = s1.innerX;
                    const col2X = s1.innerX + 90;

                    // Row 1
                    pdf.setFont("helvetica", "bold");
                    pdf.setFontSize(8);
                    pdf.setTextColor(55, 65, 81);
                    pdf.text("Name:", col1X, s1.innerY + 1);
                    pdf.setFont("helvetica", "normal");
                    pdf.setTextColor(17, 24, 39);
                    pdf.text(patientName, col1X + 28, s1.innerY + 1);

                    pdf.setFont("helvetica", "bold");
                    pdf.setTextColor(55, 65, 81);
                    pdf.text("Patient ID:", col2X, s1.innerY + 1);
                    pdf.setFont("helvetica", "normal");
                    pdf.setTextColor(17, 24, 39);
                    pdf.text(patientIdVal, col2X + 28, s1.innerY + 1);

                    // Divider 1
                    pdf.setDrawColor(241, 242, 249);
                    pdf.setLineWidth(0.2);
                    pdf.line(s1.innerX, s1.innerY + 4.5, s1.innerX + s1.innerW, s1.innerY + 4.5);

                    // Row 2
                    pdf.setFont("helvetica", "bold");
                    pdf.setTextColor(55, 65, 81);
                    pdf.text("Age / Gender:", col1X, s1.innerY + 8);
                    pdf.setFont("helvetica", "normal");
                    pdf.setTextColor(17, 24, 39);
                    pdf.text(`${patientAge} / ${patientGender}`, col1X + 28, s1.innerY + 8);

                    pdf.setFont("helvetica", "bold");
                    pdf.setTextColor(55, 65, 81);
                    pdf.text("Date of Report:", col2X, s1.innerY + 8);
                    pdf.setFont("helvetica", "normal");
                    pdf.setTextColor(17, 24, 39);
                    pdf.text(reportDate, col2X + 28, s1.innerY + 8);

                    // Divider 2
                    pdf.line(s1.innerX, s1.innerY + 11.5, s1.innerX + s1.innerW, s1.innerY + 11.5);

                    // Row 3
                    pdf.setFont("helvetica", "bold");
                    pdf.setTextColor(55, 65, 81);
                    pdf.text("Case Reference:", col1X, s1.innerY + 15);
                    pdf.setFont("helvetica", "normal");
                    pdf.setTextColor(17, 24, 39);
                    pdf.text(caseRef, col1X + 28, s1.innerY + 15);

                    pdf.setFont("helvetica", "bold");
                    pdf.setTextColor(55, 65, 81);
                    pdf.text("Analysis Time:", col2X, s1.innerY + 15);
                    pdf.setFont("helvetica", "normal");
                    pdf.setTextColor(17, 24, 39);
                    pdf.text(reportTime, col2X + 28, s1.innerY + 15);

                    currentY = s1.nextY;

                    // ==========================================
                    // SECTION 2: Image Analysis (Findings)
                    // ==========================================
                    const interpText =
                      analysisResponse.interpretation ?? "Interpretation unavailable.";
                    pdf.setFont("helvetica", "normal");
                    pdf.setFontSize(8);
                    const interpLines = pdf.splitTextToSize(interpText, s1.innerW);
                    const s2BoxH = 14 + interpLines.length * 3.6;

                    const s2 = drawSectionCard(
                      currentY,
                      "2",
                      "Image Analysis (Findings)",
                      "2  TRANSFORMATION ZONE FINDINGS",
                      s2BoxH
                    );

                    // TZ & Confidence Row
                    const tzType = analysisResponse.predicted_type ?? "Unavailable";
                    const confStr =
                      typeof analysisResponse.confidence === "number"
                        ? `${(analysisResponse.confidence * 100).toFixed(1)}%`
                        : "Unavailable";

                    pdf.setFont("helvetica", "bold");
                    pdf.setFontSize(8);
                    pdf.setTextColor(55, 65, 81);
                    pdf.text("Transformation Zone:", s2.innerX, s2.innerY + 1);
                    pdf.setFont("helvetica", "normal");
                    pdf.setTextColor(17, 24, 39);
                    pdf.text(tzType, s2.innerX + 36, s2.innerY + 1);

                    pdf.setFont("helvetica", "bold");
                    pdf.setTextColor(55, 65, 81);
                    pdf.text("Confidence:", s2.innerX + 90, s2.innerY + 1);
                    pdf.setFont("helvetica", "normal");
                    pdf.setTextColor(17, 24, 39);
                    pdf.text(confStr, s2.innerX + 115, s2.innerY + 1);

                    // Divider
                    pdf.setDrawColor(241, 242, 249);
                    pdf.setLineWidth(0.2);
                    pdf.line(s2.innerX, s2.innerY + 4.5, s2.innerX + s2.innerW, s2.innerY + 4.5);

                    // Interpretation
                    pdf.setFont("helvetica", "bold");
                    pdf.setTextColor(55, 65, 81);
                    pdf.text("Interpretation:", s2.innerX, s2.innerY + 8.5);

                    pdf.setFont("helvetica", "normal");
                    pdf.setTextColor(31, 41, 55);
                    pdf.text(interpLines, s2.innerX, s2.innerY + 12.2);

                    currentY = s2.nextY;

                    // ==========================================
                    // SECTION 3: Visualization (Grad-CAM)
                    // ==========================================
                    const gradcamResult = await generateGradCamReportImage(
                      imagePreview,
                      analysisResponse.gradcam,
                      heatmapOpacity
                    );

                    let s3BoxH = 15;
                    let imgW = 0;
                    let imgH = 0;

                    if (gradcamResult && gradcamResult.dataUrl) {
                      imgH = 43; // compact height to guarantee 1-page fit
                      imgW = (imgH * gradcamResult.width) / gradcamResult.height;
                      if (imgW > 85) {
                        imgW = 85;
                        imgH = (85 * gradcamResult.height) / gradcamResult.width;
                      }
                      s3BoxH = imgH + 8.5;
                    }

                    const s3 = drawSectionCard(
                      currentY,
                      "3",
                      "Visualization",
                      "3  GRAD-CAM ATTENTION MAP",
                      s3BoxH
                    );

                    if (gradcamResult && gradcamResult.dataUrl) {
                      const imgX = s3.innerX + (s3.innerW - imgW) / 2;
                      const imgY = s3.innerY;

                      // Subtle frame around image
                      pdf.setDrawColor(204, 232, 231);
                      pdf.setLineWidth(0.2);
                      pdf.rect(imgX - 0.4, imgY - 0.4, imgW + 0.8, imgH + 0.8, "D");

                      const imgFormat = gradcamResult.dataUrl.startsWith("data:image/png")
                        ? "PNG"
                        : "JPEG";
                      pdf.addImage(gradcamResult.dataUrl, imgFormat, imgX, imgY, imgW, imgH);

                      // Caption
                      pdf.setFont("helvetica", "normal");
                      pdf.setFontSize(7.2);
                      pdf.setTextColor(107, 114, 128);
                      pdf.text(
                        "Grad-CAM highlights regions that influenced classification; it is not a confirmed lesion or segmentation mask.",
                        s3.innerX + s3.innerW / 2,
                        imgY + imgH + 4.2,
                        { align: "center" }
                      );
                    } else {
                      pdf.setFont("helvetica", "normal");
                      pdf.setFontSize(8);
                      pdf.setTextColor(107, 114, 128);
                      pdf.text(
                        "Visualization was unavailable when this report was generated.",
                        s3.innerX,
                        s3.innerY + 5
                      );
                    }

                    currentY = s3.nextY;

                    // ==========================================
                    // SECTION 4: Assessment / Suggested Next Step
                    // ==========================================
                    let activeCentre: Facility | null = null;
                    if (selectedFacility) {
                      activeCentre = selectedFacility;
                    } else if (facilitiesUsed && facilities.length > 0) {
                      activeCentre = facilities[0];
                    } else {
                      const bgl = facilities.find(
                        (f) =>
                          f.city?.toLowerCase().includes("bengaluru") ||
                          f.city?.toLowerCase().includes("bangalore")
                      );
                      if (bgl) {
                        activeCentre = bgl;
                      } else {
                        try {
                          const controller = new AbortController();
                          const timeoutId = setTimeout(() => controller.abort(), 2000);
                          const res = await fetch(`${API_BASE}/api/facilities?city=Bengaluru`, {
                            signal: controller.signal,
                          });
                          clearTimeout(timeoutId);
                          if (res.ok) {
                            const data = await res.json();
                            const list: Facility[] = Array.isArray(data.facilities)
                              ? data.facilities
                              : [];
                            const found =
                              list.find(
                                (f) =>
                                  f.city?.toLowerCase().includes("bengaluru") ||
                                  f.city?.toLowerCase().includes("bangalore")
                              ) || list[0];
                            if (found) {
                              activeCentre = found;
                            }
                          }
                        } catch {
                          // Offline or API failure: handled by fallback 2c below
                        }
                      }
                    }

                    const nextStepVal =
                      analysisResponse.suggested_next_step ?? "Consult the examining clinician.";
                    pdf.setFont("helvetica", "normal");
                    pdf.setFontSize(8);
                    const nextStepLines = pdf.splitTextToSize(nextStepVal, s1.innerW);

                    const cleanField = (text?: string | null): string => {
                      if (!text || typeof text !== "string") return "";
                      if (text.includes("<") || text.includes(">")) return "";
                      return text.trim();
                    };

                    const referralLines: string[] = [];
                    let hasVerifiedCentre = false;
                    let refHeading = "Referral centre (Bengaluru)";

                    if (activeCentre) {
                      const cName = cleanField(activeCentre.name);
                      const cAddr = cleanField(activeCentre.address);
                      const cLocality = cleanField(
                        activeCentre.area || activeCentre.locality || activeCentre.city
                      );
                      let cFullAddr = cAddr;
                      if (cAddr && cLocality && !cAddr.toLowerCase().includes(cLocality.toLowerCase())) {
                        cFullAddr = `${cAddr}, ${cLocality}`;
                      } else if (!cAddr && cLocality) {
                        cFullAddr = cLocality;
                      }

                      if (cName || cFullAddr) {
                        hasVerifiedCentre = true;
                        if (activeCentre.city && activeCentre.city.toLowerCase() !== "bengaluru") {
                          refHeading = `Referral centre (${activeCentre.city})`;
                        } else {
                          refHeading = "Referral centre (Bengaluru)";
                        }

                        if (cName) referralLines.push(cName);
                        if (cFullAddr) {
                          const wrappedAddr = pdf.splitTextToSize(cFullAddr, s1.innerW);
                          referralLines.push(...wrappedAddr);
                        }

                        const cPhone = cleanField(activeCentre.phone);
                        if (cPhone) {
                          referralLines.push(`Phone: ${cPhone}`);
                        }

                        referralLines.push("Please verify details with the centre before visiting.");
                      }
                    }

                    const nextStepTextBottom = 4.8 + (nextStepLines.length - 1) * 3.6;
                    const refBlockStartRelY = nextStepTextBottom + 6.0;

                    let s4BoxH = 8 + nextStepLines.length * 3.6;
                    if (hasVerifiedCentre) {
                      const refDetailsStartRelY = refBlockStartRelY + 3.8;
                      const lastRefLineRelY =
                        refDetailsStartRelY + (referralLines.length - 1) * 3.6;
                      s4BoxH = Math.ceil(lastRefLineRelY + 4.0);
                    } else {
                      // Fallback 2c: single line only
                      s4BoxH = Math.ceil(refBlockStartRelY + 4.0);
                    }

                    const s4 = drawSectionCard(
                      currentY,
                      "4",
                      "Suggested Next Step",
                      "4  RECOMMENDATIONS & CLINICAL GUIDANCE",
                      s4BoxH
                    );

                    // Suggested Next Step
                    pdf.setFont("helvetica", "bold");
                    pdf.setFontSize(8);
                    pdf.setTextColor(55, 65, 81);
                    pdf.text("Suggested Next Step:", s4.innerX, s4.innerY + 1);

                    pdf.setFont("helvetica", "normal");
                    pdf.setTextColor(17, 24, 39);
                    pdf.text(nextStepLines, s4.innerX, s4.innerY + 4.8);

                    if (hasVerifiedCentre) {
                      // Referral centre heading
                      pdf.setFont("helvetica", "bold");
                      pdf.setFontSize(8);
                      pdf.setTextColor(55, 65, 81);
                      pdf.text(refHeading, s4.innerX, s4.innerY + refBlockStartRelY);

                      // Referral details
                      pdf.setFont("helvetica", "normal");
                      pdf.setFontSize(8);
                      pdf.setTextColor(17, 24, 39);
                      const refDetailsStartRelY = refBlockStartRelY + 3.8;
                      referralLines.forEach((line, idx) => {
                        pdf.text(line, s4.innerX, s4.innerY + refDetailsStartRelY + idx * 3.6);
                      });
                    } else {
                      // Fallback 2c
                      pdf.setFont("helvetica", "normal");
                      pdf.setFontSize(8);
                      pdf.setTextColor(17, 24, 39);
                      pdf.text(
                        "Referral centre: please consult your examining clinician.",
                        s4.innerX,
                        s4.innerY + refBlockStartRelY
                      );
                    }

                    // Draw footer on final page
                    drawFooter();

                    setReportSaveStatus("");
                    if (savedScan?.id && patientId && patientAccessToken) {
                      try {
                        const reportBlob = pdf.output("blob");
                        let saveResponse: Response;
                        if (import.meta.env.PROD) {
                          const grantResponse = await fetch(
                            `${API_BASE}/api/patients/${encodeURIComponent(patientId)}/upload-grant`,
                            {
                              method: "POST",
                              headers: {
                                "Content-Type": "application/json",
                                "X-Patient-Access-Token": patientAccessToken,
                              },
                              body: JSON.stringify({
                                scan_id: savedScan.id,
                                content_type: "application/pdf",
                                kind: "report",
                              }),
                            },
                          );
                          const grant = await grantResponse.json();
                          if (!grantResponse.ok) throw new Error(getErrorMessage(grant));
                          const { upload } = await import("@vercel/blob/client");
                          const uploaded = await upload(grant.pathname, reportBlob, {
                            access: "private",
                            contentType: "application/pdf",
                            handleUploadUrl: `${API_BASE}/api/blob-upload`,
                            clientPayload: grant.upload_grant,
                            multipart: true,
                          });
                          if (uploaded.pathname !== grant.pathname) {
                            throw new Error("The private report upload returned an unexpected path.");
                          }
                          saveResponse = await fetch(`${API_BASE}/api/scans/${encodeURIComponent(savedScan.id)}/report`, {
                            method: "POST",
                            headers: {
                              "Content-Type": "application/json",
                              "X-Patient-Access-Token": patientAccessToken,
                            },
                            body: JSON.stringify({ blob_pathname: uploaded.pathname }),
                          });
                        } else {
                          saveResponse = await fetch(`${API_BASE}/api/scans/${encodeURIComponent(savedScan.id)}/report`, {
                            method: "POST",
                            headers: {
                              "Content-Type": "application/pdf",
                              "X-Patient-Access-Token": patientAccessToken,
                            },
                            body: reportBlob,
                          });
                        }
                        const saveData = await saveResponse.json().catch(() => ({}));
                        if (!saveResponse.ok) {
                          throw new Error(saveData.detail || "The report could not be saved to private storage.");
                        }
                        setReportSaveStatus("A private copy of this report was saved with the scan.");
                      } catch (error) {
                        setReportSaveStatus(
                          error instanceof Error
                            ? `The PDF was downloaded, but its private copy could not be saved: ${error.message}`
                            : "The PDF was downloaded, but its private copy could not be saved.",
                        );
                      }
                    } else {
                      setReportSaveStatus("The PDF was downloaded. A private copy could not be linked to this scan.");
                    }

                    pdf.save("cervical-analysis-report.pdf");
                  } catch (err) {
                    console.error("Failed to generate PDF report:", err);
                  } finally {
                    setPdfLoading(false);
                  }
                }}
              >
                {pdfLoading ? "Generating Report..." : "Download PDF Report"}
              </button>
              {reportSaveStatus && (
                <p style={{ margin: "8px 0 0", fontSize: "12px", color: "#4b5f63" }} role="status">
                  {reportSaveStatus}
                </p>
              )}



              <button
                type="button"
                className="primary-button"
                onClick={() => {
                  setAnalysisResponse(null);
                  setSavedScan(null);
                  setUploadError("");
                  setStep("upload");
                }}
              >
                <RotateCcw size={17} />
                Analyze another image
              </button>

              <div className="upload-bottom-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={startAnotherPatient}
                >
                  <UserRound size={17} />
                  New patient
                </button>
              </div>

            </>
          )}
        </section>
      </section>

      {aboutOpen && (
        <div
          className="modal-backdrop"
          onClick={() => setAboutOpen(false)}
          role="presentation"
        >
          <section
            className="about-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="about-title"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              type="button"
              className="modal-close"
              onClick={() => setAboutOpen(false)}
              aria-label="Close about dialog"
            >
              <X size={20} />
            </button>
            <div className="modal-icon">
              <Activity size={26} />
            </div>
            <p className="eyebrow">ABOUT THE PROJECT</p>
            <h2 id="about-title">Early Cervical Risk Detection</h2>
            <p>
              This research prototype explores AI-assisted analysis of
              colposcopy images, transformation-zone classification, and
              Grad-CAM visual explanations.
            </p>
            <p>
              The prototype is intended for research and education. It is not
              a diagnostic tool and must not replace assessment by a qualified
              healthcare professional.
            </p>
            <button
              type="button"
              className="primary-button"
              onClick={() => setAboutOpen(false)}
            >
              Close
            </button>
          </section>
        </div>
      )}

      {historyOpen && (
        <div
          className="modal-backdrop"
          onClick={() => setHistoryOpen(false)}
          role="presentation"
        >
          <section
            className="history-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="history-title"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              type="button"
              className="modal-close"
              onClick={() => setHistoryOpen(false)}
              aria-label="Close scan history"
            >
              <X size={20} />
            </button>
            <div className="modal-icon"><FileText size={24} /></div>
            <p className="eyebrow">PRIVATE PATIENT RECORD</p>
            <h2 id="history-title">View previous cases</h2>
            <p className="history-intro">
              Enter the patient ID and access code saved when the record was created. The code is not stored in this browser.
            </p>
            <form className="history-lookup-form" onSubmit={handleHistorySubmit}>
              <label htmlFor="history-patient-id">Patient ID</label>
              <input
                id="history-patient-id"
                value={historyPatientId}
                onChange={(event) => setHistoryPatientId(event.target.value)}
                autoComplete="off"
                required
              />
              <label htmlFor="history-access-code">Access code</label>
              <input
                id="history-access-code"
                type="password"
                value={historyAccessToken}
                onChange={(event) => setHistoryAccessToken(event.target.value)}
                autoComplete="off"
                required
              />
              {historyError && <div className="error-message" role="alert">{historyError}</div>}
              <button
                className="primary-button"
                type="submit"
                disabled={historyLoading || !historyPatientId.trim() || !historyAccessToken.trim()}
              >
                {historyLoading ? <><LoaderCircle className="spin" size={18} /> Loading history...</> : "Load scan history"}
              </button>
            </form>
            {historyPatient && historyScans && (
              <div className="history-results" aria-live="polite">
                <h3>{historyPatient.name} <span>({historyScans.length} scans)</span></h3>
                {historyScans.length === 0 ? (
                  <p className="history-empty">No scans are linked to this patient yet.</p>
                ) : historyScans.map((scan) => (
                  <article className="history-scan-card" key={scan.id}>
                    <div className="history-scan-heading">
                      <strong>{scan.predicted_type || "Image not accepted"}</strong>
                      <span>{scan.created_at ? new Date(scan.created_at).toLocaleString() : "Date unavailable"}</span>
                    </div>
                    <p>Confidence: {typeof scan.confidence === "number" ? `${(scan.confidence * 100).toFixed(1)}%` : "Not available"}</p>
                    {scan.prediction.interpretation && <p>{scan.prediction.interpretation}</p>}
                    {scan.prediction.suggested_next_step && <p>{scan.prediction.suggested_next_step}</p>}
                    {scan.has_report && (
                      <button type="button" className="patient-details-btn" onClick={() => void downloadHistoryReport(scan.id)}>
                        Download saved report
                      </button>
                    )}
                  </article>
                ))}
              </div>
            )}
          </section>
        </div>
      )}

      {facilitiesOpen && (
        <div
          className="modal-backdrop"
          onClick={() => setFacilitiesOpen(false)}
          role="presentation"
        >
          <section
            className="facilities-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="facilities-title"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              type="button"
              className="modal-close"
              onClick={() => setFacilitiesOpen(false)}
              aria-label="Close care centres modal"
            >
              <X size={18} />
            </button>

            <div className="facilities-header">
              <div className="modal-icon">
                <MapPin size={24} />
              </div>
              <h2 id="facilities-title">Nearby Referral &amp; Care Centres</h2>
              <p>
                Verified tertiary referral hospitals, gynaecologic oncology
                clinics, and screening centers.
              </p>
            </div>

            <form
              className="facilities-search-bar"
              onSubmit={(e) => {
                e.preventDefault();
                fetchFacilities(facilitiesQuery);
              }}
            >
              <input
                type="text"
                placeholder="Search by city (e.g. Delhi, Mumbai) or PIN code..."
                value={facilitiesQuery}
                onChange={(e) => setFacilitiesQuery(e.target.value)}
              />
              <button type="submit" className="facilities-search-btn">
                Search
              </button>
              <button
                type="button"
                className="facilities-geo-btn"
                onClick={handleUseLocation}
                title="Sort by proximity using your current location"
              >
                <MapPin size={14} /> Near me
              </button>
            </form>

            {facilitiesError && (
              <div className="modal-notice" style={{ margin: "0 0 12px" }}>
                {facilitiesError}
              </div>
            )}

            {facilitiesLoading ? (
              <div
                style={{
                  padding: "30px",
                  textAlign: "center",
                  color: "#4b5f63",
                }}
              >
                <LoaderCircle
                  className="spin"
                  size={24}
                  style={{ margin: "0 auto 8px" }}
                />
                <span>Locating care centres...</span>
              </div>
            ) : facilities.length === 0 ? (
              <div
                style={{
                  padding: "24px",
                  textAlign: "center",
                  color: "#607274",
                }}
              >
                No verified centres matched your search. Try another city or
                clear the search filter.
              </div>
            ) : (
              <div className="facilities-list">
                {facilities.map((fac) => (
                  <div
                    className="facility-card"
                    key={fac.id}
                    onClick={() => setSelectedFacility(fac)}
                  >
                    <div className="facility-top">
                      <span className="facility-name">{fac.name}</span>
                      {typeof fac.distance_km === "number" && (
                        <span className="facility-distance-badge">
                          {fac.distance_km} km away
                        </span>
                      )}
                    </div>
                    <span className="facility-type">
                      {fac.type} · {fac.city}, {fac.state}
                    </span>
                    <span className="facility-address">
                      {fac.address}{" "}
                      {fac.pincode && `— PIN: ${fac.pincode}`}
                    </span>
                    <div className="facility-specialities">
                      {fac.specialities.map((spec) => (
                        <span className="facility-speciality-pill" key={spec}>
                          {spec}
                        </span>
                      ))}
                    </div>
                    <div className="facility-contacts">
                      {fac.phone && (
                        <a href={`tel:${fac.phone}`}>
                          <Phone size={13} /> {fac.phone}
                        </a>
                      )}
                      {fac.website && (
                        <a
                          href={fac.website}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <ExternalLink size={13} /> Official Website
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {facilitiesNote && (
              <div className="facility-note-footer">
                <strong>Note:</strong> {facilitiesNote}
              </div>
            )}
          </section>
        </div>
      )}
    </main>
  );
}
