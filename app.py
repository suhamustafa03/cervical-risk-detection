"""Streamlit frontend for the Cervical AI Analysis research prototype.

Requires the FastAPI backend from the project upgrade to be running at
http://127.0.0.1:8000. This file does not retrain or change either model.
"""
import hashlib
import os

import cv2
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("CERVICAL_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Cervical AI Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root { --navy:#17324d; --teal:#147d73; --muted:#627487; }
    .stApp { background:linear-gradient(180deg,#f4f8fc 0%,#fbfcfe 100%); }
    .block-container { max-width:1250px; padding-top:1.2rem; padding-bottom:2.5rem; }
    h1,h2,h3,p,label { font-family:Inter,system-ui,sans-serif; }
    h1 { color:var(--navy); font-weight:800!important; letter-spacing:-.035em; }
    h2 { color:var(--navy); font-weight:750!important; margin-top:1rem!important; }
    h3 { color:var(--navy); font-weight:700!important; }
    p, label { line-height:1.6; }
    [data-testid="stVerticalBlockBorderWrapper"] { background:white; border:1px solid #e0e8f0; border-radius:16px; box-shadow:0 5px 18px rgba(23,50,77,.045); }
    [data-testid="stMetric"] { background:#fff; border:1px solid #e0e8f0; border-radius:14px; padding:14px 16px; }
    [data-testid="stMetricValue"] { color:var(--navy); font-size:1.7rem!important; font-weight:800!important; }
    [data-testid="stFileUploader"] { background:white; border-radius:14px; padding:8px; }
    .small-muted { color:#627487; font-size:.94rem; }
    .result-hero { padding:1.35rem 1.5rem; border-radius:18px; background:linear-gradient(120deg,#e9f7f4,#f5fbff); border:1px solid #bfe5dc; }
    .result-kicker { text-transform:uppercase; letter-spacing:.12em; color:#55716f; font-size:.78rem; font-weight:700; }
    .result-type { font-size:clamp(2.6rem,5vw,4rem); color:#147d73; font-weight:800; line-height:1.12; margin:.35rem 0; }
    .clinical-note { background:#fff8e8; border:1px solid #f2dfae; border-radius:14px; padding:1rem 1.15rem; color:#665a36; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Cervical AI Analysis")
st.markdown("**Explainable AI-assisted transformation-zone analysis**")
st.caption("Research prototype • Image screening, classification and visual explanation")
st.divider()

with st.container(border=True):
    st.subheader("Upload a colposcopy image")
    st.write("Choose a JPG or PNG image to run the analysis.")
    uploaded = st.file_uploader(
        "Select image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    run_analysis = st.button(
        "Analyse image",
        type="primary",
        disabled=uploaded is None,
        width="stretch",
    )

if uploaded is None:
    st.info("Upload an image to begin. The result is a research aid and is not a medical diagnosis.")
    st.stop()

raw_image = uploaded.getvalue()
try:
    image = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("This file could not be opened as an image. Please choose a valid JPG or PNG file.")
    st.stop()

# Clear a previous prediction whenever the uploaded file changes.
upload_signature = hashlib.sha256(raw_image).hexdigest()
if st.session_state.get("upload_signature") != upload_signature:
    st.session_state["upload_signature"] = upload_signature
    st.session_state.pop("analysis_result", None)

if run_analysis:
    with st.spinner("Validating image and running AI analysis…"):
        try:
            response = requests.post(
                f"{API_URL}/analyze",
                files={
                    "file": (
                        uploaded.name,
                        raw_image,
                        uploaded.type or "image/jpeg",
                    )
                },
                timeout=180,
            )
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                st.error(f"Analysis service error: {detail}")
                st.caption("Check that the FastAPI backend is running in the other Terminal tab.")
                st.stop()
            st.session_state["analysis_result"] = response.json()
        except requests.exceptions.RequestException as exc:
            st.error("The backend could not be reached. Start the API in the other Terminal tab, then try again.")
            st.code("uvicorn api:app --host 127.0.0.1 --port 8000")
            st.caption(f"Connection detail: {exc}")
            st.stop()

if "analysis_result" not in st.session_state:
    st.image(image, caption="Selected image", width="stretch")
    st.caption("Select **Analyse image** when you are ready.")
    st.stop()

result = st.session_state["analysis_result"]

if not result.get("valid_cervical_image"):
    st.error(result.get("message", "The image did not pass input screening."))
    st.caption("Input screening is preliminary and cannot guarantee rejection of every unrelated image.")
    st.stop()

# Values returned by the backend for this exact uploaded image.
predicted = result["predicted_type"]
probabilities = result["probabilities"]
cam = np.asarray(result["gradcam"], dtype=np.float32)
cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)
cam = np.clip(cam, 0.0, 1.0)

st.header("AI analysis result")
with st.container(border=True):
    st.markdown(
        f"""<div class="result-hero">
        <div class="result-kicker">Predicted transformation-zone type</div>
        <div class="result-type">{predicted}</div>
        <div><b>Model confidence: {result['confidence'] * 100:.1f}%</b></div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption("Confidence is the model's relative score for this class, not the probability that cancer is present.")

st.subheader("What this means")
st.write(result["interpretation"])
probability_columns = st.columns(3)
for col, key in zip(probability_columns, ["Type 1", "Type 2", "Type 3"]):
    with col:
        st.metric(key, f"{probabilities[key] * 100:.1f}%")

st.header("Image explanation")
st.write(
    "Grad-CAM highlights image regions that influenced the classification. "
    "The heatmap and attention surface do not identify or confirm a lesion."
)
heat = cv2.resize(cam, image.size, interpolation=cv2.INTER_CUBIC)
heat = np.clip(heat, 0.0, 1.0)
original = np.asarray(image)
heat_color = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET)
heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(original, 0.55, heat_color, 0.45, 0)
original_col, overlay_col = st.columns(2)
with original_col:
    st.image(image, caption="Original colposcopy image", width="stretch")
with overlay_col:
    st.image(overlay, caption="Grad-CAM overlay", width="stretch")

# One common Grad-CAM-derived surface feeds both visualizers.
st.header("3D attention visualisation")
st.write(
    "Explore the activation map in two views. Both visualizers use the same Grad-CAM map "
    "generated for the uploaded image and its predicted class. These are 3D plots of model "
    "attention, not a 3D reconstruction of the cervix or a tumour."
)

control_col1, control_col2 = st.columns(2)
with control_col1:
    height_scale = st.slider(
        "Surface height",
        min_value=0.2,
        max_value=3.0,
        value=1.0,
        step=0.1,
        key="attention_height",
    )
with control_col2:
    resolution = st.slider(
        "Surface detail",
        min_value=20,
        max_value=60,
        value=40,
        step=10,
        key="attention_resolution",
    )

surface = cv2.resize(cam, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
surface = np.clip(surface, 0.0, 1.0) * height_scale
x = np.linspace(0.0, 1.0, resolution)
y = np.linspace(0.0, 1.0, resolution)

st.subheader("View 1 — Horizontal 3D attention surface")
st.caption("Wide landscape view for examining the distribution of activation across the image.")
horizontal_fig = go.Figure(
    data=[
        go.Surface(
            x=x,
            y=y,
            z=surface,
            surfacecolor=surface,
            colorscale="Turbo",
            colorbar={"title": "Activation", "len": 0.72},
            hovertemplate=(
                "Image width: %{x:.2f}<br>"
                "Image height: %{y:.2f}<br>"
                "Activation: %{z:.2f}<extra></extra>"
            ),
        )
    ]
)
horizontal_fig.update_layout(
    height=440,
    autosize=True,
    margin=dict(l=5, r=5, t=10, b=5),
    scene=dict(
        xaxis_title="Image width",
        yaxis_title="Image height",
        zaxis_title="Attention strength",
        aspectmode="manual",
        aspectratio=dict(x=1.8, y=1.0, z=0.65),
        camera=dict(eye=dict(x=1.7, y=1.5, z=1.0)),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(
    horizontal_fig,
    width="stretch",
    config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    key="horizontal_attention_surface",
)

st.subheader("View 2 — Interactive 3D attention surface")
st.caption("Drag to rotate and use the scroll gesture to zoom. The same activation map is used above and below.")
interactive_fig = go.Figure(
    data=[
        go.Surface(
            x=x,
            y=y,
            z=surface,
            surfacecolor=surface,
            colorscale="Turbo",
            colorbar={"title": "Activation", "len": 0.72},
            hovertemplate=(
                "Image width: %{x:.2f}<br>"
                "Image height: %{y:.2f}<br>"
                "Attention strength: %{z:.2f}<extra></extra>"
            ),
        )
    ]
)
interactive_fig.update_layout(
    height=520,
    autosize=True,
    margin=dict(l=5, r=5, t=10, b=5),
    scene=dict(
        xaxis_title="Image width",
        yaxis_title="Image height",
        zaxis_title="Attention strength",
        aspectmode="cube",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.15)),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(
    interactive_fig,
    width="stretch",
    config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    key="interactive_attention_surface",
)

st.header("Analysis summary")
with st.container(border=True):
    st.markdown(f"**AI result:** {predicted}")
    st.markdown(f"**Interpretation:** {result['interpretation']}")
    st.markdown(f"**Suggested next step:** {result['suggested_next_step']}")

st.header("Clinical follow-up")
st.write(
    "Discuss this result with your gynecologist or the clinician who performed the examination. "
    "They can interpret it alongside your full examination and any other test results."
)

st.markdown(
    "<div class='clinical-note'><b>Important clinical note</b><br>"
    "Type 1, Type 2 and Type 3 describe transformation-zone anatomy and visibility. "
    "They are not cancer stages and do not directly indicate whether cancer or a lesion is present. "
    "Grad-CAM and its 3D visualizations show model attention, not confirmed abnormal tissue.</div>",
    unsafe_allow_html=True,
)
st.warning(result["disclaimer"])
st.caption(
    "Research prototype only. Do not use this system to decide on treatment or to delay professional medical care."
)
