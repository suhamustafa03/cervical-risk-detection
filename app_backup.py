"""Streamlit frontend for the Cervical AI Analysis API."""
import os
import requests
import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
from PIL import Image

API_URL = os.getenv("CERVICAL_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Cervical AI Analysis", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --navy:#17324d; --teal:#147d73; --muted:#627487; }
.stApp { background:linear-gradient(180deg,#f4f8fc 0%,#fbfcfe 100%); }
.block-container { max-width:1200px; padding-top:1.1rem; padding-bottom:2.5rem; }
h1,h2,h3,p,label,div { font-family:Inter,system-ui,sans-serif; }
h1 { color:var(--navy); font-weight:800!important; letter-spacing:-.035em; }
h2 { color:var(--navy); font-weight:750!important; margin-top:1rem!important; }
h3 { color:var(--navy); font-weight:700!important; }
p, label { line-height:1.6; }
[data-testid="stVerticalBlockBorderWrapper"] { background:white; border:1px solid #e0e8f0; border-radius:16px; box-shadow:0 5px 18px rgba(23,50,77,.045); }
[data-testid="stMetric"] { background:#fff; border:1px solid #e0e8f0; border-radius:14px; padding:14px 16px; }
[data-testid="stMetricValue"] { color:var(--navy); font-size:1.7rem!important; font-weight:800!important; }
[data-testid="stFileUploader"] { background:white; border-radius:14px; padding:8px; }
.small-muted { color:#627487; font-size:.94rem; }
.result-hero { padding:1.2rem 1.4rem; border-radius:18px; background:linear-gradient(120deg,#e9f7f4,#f5fbff); border:1px solid #bfe5dc; }
.result-kicker { text-transform:uppercase; letter-spacing:.12em; color:#55716f; font-size:.78rem; font-weight:700; }
.result-type { font-size:clamp(2.6rem,5vw,4rem); color:#147d73; font-weight:800; line-height:1.12; margin:.3rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Cervical AI Analysis")
st.markdown("**Explainable AI-assisted transformation-zone analysis**")
st.caption("Research prototype • Image screening, classification and visual explanation")
st.divider()

with st.container(border=True):
    st.subheader("Upload a colposcopy image")
    st.write("Choose a JPG or PNG image to run the analysis.")
    uploaded = st.file_uploader("Select image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    run = st.button("Analyse image", type="primary", disabled=uploaded is None, use_container_width=True)

if uploaded is None:
    st.info("Upload an image to begin. The result is a research aid and is not a medical diagnosis.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("This file could not be opened as an image. Please choose a JPG or PNG file.")
    st.stop()

# Keep the latest analysis visible when users adjust the 3D viewer controls.
import hashlib
upload_signature = hashlib.sha256(uploaded.getvalue()).hexdigest()
if st.session_state.get("upload_signature") != upload_signature:
    st.session_state["upload_signature"] = upload_signature
    st.session_state.pop("analysis_result", None)

if run:
    with st.spinner("Validating image and running AI analysis…"):
        try:
            response = requests.post(
                f"{API_URL}/analyze",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")},
                timeout=180,
            )
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                st.error(f"Analysis service error: {detail}")
                st.caption("Check that the FastAPI backend is running.")
                st.stop()
            st.session_state["analysis_result"] = response.json()
        except requests.exceptions.RequestException as exc:
            st.error("The backend is not reachable. Start the API in a second Terminal window, then try again.")
            st.code("uvicorn api:app --host 127.0.0.1 --port 8000")
            st.caption(f"Connection detail: {exc}")
            st.stop()

if "analysis_result" not in st.session_state:
    st.image(image, caption="Selected image", use_container_width=True)
    st.caption("Select **Analyse image** when you are ready.")
    st.stop()

result = st.session_state["analysis_result"]

if not result.get("valid_cervical_image"):
    st.error(result.get("message", "The image did not pass input screening."))
    st.caption("Input screening is preliminary and cannot guarantee detection of every unrelated image.")
    st.stop()

predicted = result["predicted_type"]
probabilities = result["probabilities"]
cam = np.array(result["gradcam"], dtype=np.float32)

st.header("Analysis result")
with st.container(border=True):
    st.markdown(f'''<div class="result-hero"><div class="result-kicker">Predicted transformation-zone type</div><div class="result-type">{predicted}</div><div><b>Model confidence: {result['confidence']*100:.1f}%</b></div></div>''', unsafe_allow_html=True)
    st.caption("Confidence is the model's relative score for this class, not the probability that cancer is present.")

st.subheader("What this means")
st.write(result["interpretation"])
cols = st.columns(3)
for col, key in zip(cols, ["Type 1", "Type 2", "Type 3"]):
    with col:
        st.metric(key, f"{probabilities[key]*100:.1f}%")

st.header("Image explanation")
st.write("Grad-CAM highlights image regions that influenced the classification. It does not identify or confirm a lesion.")
heat = cv2.resize(cam, image.size, interpolation=cv2.INTER_CUBIC)
heat = np.clip(heat, 0, 1)
original = np.asarray(image)
heat_color = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_JET)
heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(original, .55, heat_color, .45, 0)
left, right = st.columns(2)
with left:
    st.image(image, caption="Original image", use_container_width=True)
with right:
    st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

st.subheader("Interactive 3D attention surface")
st.write("The surface height represents Grad-CAM activation strength. Rotate by dragging; zoom with the scroll gesture. This is an attention visualization, not a 3D scan or lesion reconstruction.")
height_scale = st.slider("Surface height", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
resolution = st.slider("Surface detail", min_value=20, max_value=60, value=40, step=10)
surface = cv2.resize(cam, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
surface = np.clip(surface, 0, 1) * height_scale
x = np.linspace(0, 1, resolution)
y = np.linspace(0, 1, resolution)
fig = go.Figure(data=[go.Surface(x=x, y=y, z=surface, surfacecolor=surface, colorscale="Turbo", colorbar={"title": "Activation", "len": 0.7}, hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Activation: %{z:.2f}<extra></extra>")])
fig.update_layout(
    height=520, margin=dict(l=0, r=0, t=20, b=0),
    scene=dict(xaxis_title="Image width", yaxis_title="Image height", zaxis_title="Attention strength", aspectmode="cube", camera=dict(eye=dict(x=1.5,y=1.5,z=1.15))),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

st.header("Analysis summary")
with st.container(border=True):
    st.markdown(f"**AI result:** {predicted}")
    st.markdown(f"**Interpretation:** {result['interpretation']}")
    st.markdown(f"**Suggested next step:** {result['suggested_next_step']}")

st.header("Clinical follow-up")
st.write("For a real appointment, contact your own gynecologist or the hospital/colposcopy clinic that performed the examination. They can advise whether further assessment or biopsy is appropriate. No contact details are shown here because unverified sample contacts could misdirect patients.")

st.warning(result["disclaimer"])
st.caption("Do not use this prototype to decide on treatment or to delay clinical care. Type 1, Type 2 and Type 3 describe transformation-zone anatomy and visibility; they are not cancer stages.")
