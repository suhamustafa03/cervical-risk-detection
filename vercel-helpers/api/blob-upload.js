import { createHmac, timingSafeEqual } from "node:crypto";
import { handleUpload } from "@vercel/blob/client";

const IMAGE_MAX = 20 * 1024 * 1024;
const REPORT_MAX = 20 * 1024 * 1024;
const MIME_TYPES = new Set(["image/jpeg", "image/png", "application/pdf"]);

function verifyGrant(grant, pathname) {
  const secret = process.env.UPLOAD_SIGNING_SECRET || "";
  if (secret.length < 32 || typeof grant !== "string") return null;
  const [encoded, suppliedText, extra] = grant.split(".");
  if (!encoded || !suppliedText || extra !== undefined) return null;
  try {
    const supplied = Buffer.from(suppliedText, "base64url");
    const expected = createHmac("sha256", secret).update(encoded).digest();
    if (supplied.length !== expected.length || !timingSafeEqual(supplied, expected)) return null;
    const payload = JSON.parse(Buffer.from(encoded, "base64url").toString("utf8"));
    if (payload.pathname !== pathname || !MIME_TYPES.has(payload.content_type)) return null;
    if (payload.expires_at * 1000 < Date.now()) return null;
    const maxForType = payload.content_type === "application/pdf" ? REPORT_MAX : IMAGE_MAX;
    if (payload.maximum_size !== maxForType) return null;
    const pathIsImage = /^patients\/[0-9a-f-]{36}\/scans\/[0-9a-f-]{36}\/original\.(?:jpg|png)$/i.test(pathname);
    const pathIsReport = /^patients\/[0-9a-f-]{36}\/scans\/[0-9a-f-]{36}\/report\.pdf$/i.test(pathname);
    if (payload.kind !== (pathIsReport ? "report" : "image") ||
        (payload.content_type === "application/pdf") !== pathIsReport ||
        (payload.content_type !== "application/pdf") !== pathIsImage) return null;
    return payload;
  } catch {
    return null;
  }
}

export default async function handler(request, response) {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return response.status(405).json({ error: "Method not allowed" });
  }

  try {
    const body = typeof request.body === "string" ? JSON.parse(request.body) : request.body;
    const action = body?.type;
    const pathname = body?.payload?.pathname;
    const clientPayload = body?.payload?.clientPayload;
    if (action !== "blob.generate-client-token" && action !== "blob.upload-completed") {
      return response.status(400).json({ error: "Unsupported upload action" });
    }
    if (action === "blob.generate-client-token" && !verifyGrant(clientPayload, pathname)) {
      return response.status(401).json({ error: "Upload grant is invalid or expired" });
    }

    const result = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (requestedPath, requestedGrant) => {
        const grant = verifyGrant(requestedGrant, requestedPath);
        if (!grant) throw new Error("Upload grant is invalid or expired");
        return {
          addRandomSuffix: false,
          allowedContentTypes: [grant.content_type],
          maximumSizeInBytes: grant.maximum_size,
          validUntil: grant.expires_at * 1000,
          allowOverwrite: grant.kind === "report",
        };
      },
      onUploadCompleted: async () => undefined,
    });
    return response.status(200).json(result);
  } catch {
    return response.status(400).json({ error: "Private upload could not be authorized" });
  }
}
