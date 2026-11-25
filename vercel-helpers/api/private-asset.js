import { issueSignedToken, presignUrl } from "@vercel/blob";

const ASSET_KINDS = new Set(["original", "overlay", "heatmap", "report"]);

export default async function handler(request, response) {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return response.status(405).json({ error: "Method not allowed" });
  }

  const backendBase = process.env.BACKEND_INTERNAL_URL;
  const payload = request.body;
  const scanId = typeof payload?.scan_id === "string" ? payload.scan_id : "";
  const kind = typeof payload?.kind === "string" ? payload.kind : "";
  const patientToken = request.headers["x-patient-access-token"];
  if (!backendBase || !/^[0-9a-f-]{36}$/i.test(scanId) || !ASSET_KINDS.has(kind) ||
      typeof patientToken !== "string" || !patientToken) {
    return response.status(400).json({ error: "A scan, asset type, and patient access token are required" });
  }

  try {
    const backendUrl = new URL(
      `/api/scans/${encodeURIComponent(scanId)}/assets/${encodeURIComponent(kind)}`,
      backendBase,
    );
    const assetResponse = await fetch(backendUrl, {
      headers: { "x-patient-access-token": patientToken },
    });
    const asset = await assetResponse.json();
    if (!assetResponse.ok) {
      return response.status(assetResponse.status).json({ error: asset.detail || "Asset is unavailable" });
    }
    if (typeof asset.blob_pathname !== "string" || !asset.blob_pathname.startsWith("patients/")) {
      return response.status(404).json({ error: "Asset is not in private object storage" });
    }

    const expiresAt = Date.now() + 5 * 60 * 1000;
    const signedToken = await issueSignedToken({
      pathname: asset.blob_pathname,
      operations: ["get"],
      validUntil: expiresAt,
    });
    const signedUrl = await presignUrl(signedToken, {
      pathname: asset.blob_pathname,
      operation: "get",
      validUntil: expiresAt,
    });
    response.setHeader("Cache-Control", "no-store");
    return response.status(200).json({ url: signedUrl.presignedUrl, expires_at: expiresAt });
  } catch {
    return response.status(502).json({ error: "Could not create a temporary private asset URL" });
  }
}
