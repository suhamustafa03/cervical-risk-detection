import express from "express";
import blobUploadHandler from "./api/blob-upload.js";
import privateAssetHandler from "./api/private-asset.js";

const app = express();
app.disable("x-powered-by");
app.use(express.json({ limit: "1mb" }));
app.all("/api/blob-upload", blobUploadHandler);
app.all("/api/private-asset", privateAssetHandler);

app.use((error, _request, response, _next) => {
  if (response.headersSent) return;
  const status = error?.status === 413 ? 413 : 400;
  return response.status(status).json({ error: status === 413 ? "Request payload is too large" : "Invalid request" });
});

export default app;
