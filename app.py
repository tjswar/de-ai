import streamlit as st
from PIL import Image
from transformers import pipeline
import torch

st.set_page_config(page_title="AI Image Detector", layout="wide")
st.title("AI Image Detector")
st.caption("Upload or capture images and predict: REAL vs AI generated (FAKE). Use as a triage signal, not final proof.")

MODEL_NAME = "dima806/ai_vs_real_image_detection"

@st.cache_resource
def load_pipe():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-classification", model=MODEL_NAME, device=device)

pipe = load_pipe()

def analyze_one(img: Image.Image):
    preds = pipe(img)
    scores = {p["label"].upper(): float(p["score"]) for p in preds}
    real = scores.get("REAL", 0.0)
    fake = scores.get("FAKE", 0.0)
    return preds, real, fake

def verdict_label(fake: float, threshold: float, gray_zone: float):
    # gray_zone applies around the threshold: [t-g, t+g]
    if abs(fake - threshold) <= gray_zone:
        return "Manual review", "⚠️"
    if fake >= threshold:
        return "AI generated (FAKE)", "🛑"
    return "Likely real", "✅"

st.sidebar.header("Controls")
threshold = st.sidebar.slider(
    "Flag as AI generated if FAKE confidence ≥",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01
)
gray_zone = st.sidebar.slider(
    "Manual review zone (± around threshold)",
    min_value=0.00,
    max_value=0.30,
    value=0.05,
    step=0.01,
    help="If FAKE confidence is close to the threshold, label it for manual review."
)

st.sidebar.divider()
mode = st.sidebar.radio("Input mode", ["Single image", "Batch (multiple images)"], index=0)

if mode == "Single image":
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Upload")
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"], key="single_upload")

    with colB:
        st.subheader("Camera")
        cam = st.camera_input("Take a photo", key="single_cam")

    data_img = None
    if uploaded is not None:
        data_img = Image.open(uploaded).convert("RGB")
    elif cam is not None:
        data_img = Image.open(cam).convert("RGB")

    if data_img is not None:
        left, right = st.columns([1.2, 1])

        with left:
            st.image(data_img, caption="Input image", use_container_width=True)

        with right:
            with st.spinner("Analyzing..."):
                preds, real, fake = analyze_one(data_img)

            label, icon = verdict_label(fake, threshold, gray_zone)

            st.subheader("Result")
            st.write(f"**Verdict:** {icon} {label}")
            st.metric("AI generated (FAKE) confidence", f"{fake:.3f}")
            st.progress(min(max(fake, 0.0), 1.0))

            st.caption(f"REAL confidence: {real:.3f}")

            with st.expander("Raw model output"):
                st.json(preds)

else:
    st.subheader("Batch analysis")
    files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if files:
        st.info(f"{len(files)} image(s) uploaded. Analyzing...")

        results = []
        for f in files:
            try:
                img = Image.open(f).convert("RGB")
                preds, real, fake = analyze_one(img)
                label, icon = verdict_label(fake, threshold, gray_zone)
                results.append({
                    "filename": f.name,
                    "fake_confidence": fake,
                    "real_confidence": real,
                    "verdict": f"{icon} {label}",
                    "raw": preds,
                    "image": img
                })
            except Exception as e:
                results.append({
                    "filename": getattr(f, "name", "unknown"),
                    "fake_confidence": 0.0,
                    "real_confidence": 0.0,
                    "verdict": f"❌ Error: {e}",
                    "raw": [],
                    "image": None
                })

        sort_key = st.selectbox(
            "Sort by",
            ["Highest FAKE confidence", "Lowest FAKE confidence", "Filename A→Z"],
            index=0
        )

        if sort_key == "Highest FAKE confidence":
            results.sort(key=lambda x: x["fake_confidence"], reverse=True)
        elif sort_key == "Lowest FAKE confidence":
            results.sort(key=lambda x: x["fake_confidence"])
        else:
            results.sort(key=lambda x: x["filename"].lower())

        st.divider()
        for r in results:
            with st.container(border=True):
                c1, c2 = st.columns([1, 2])

                with c1:
                    if r["image"] is not None:
                        st.image(r["image"], use_container_width=True)
                    else:
                        st.write("No preview")

                with c2:
                    st.write(f"**File:** {r['filename']}")
                    st.write(f"**Verdict:** {r['verdict']}")
                    st.write(f"**FAKE:** {r['fake_confidence']:.3f}    |    **REAL:** {r['real_confidence']:.3f}")
                    st.progress(min(max(r["fake_confidence"], 0.0), 1.0))
                    with st.expander("Raw model output"):
                        st.json(r["raw"])

st.divider()
st.caption(
    "Tip: If you see lots of false positives or false negatives, adjust the threshold and manual review zone. "
    "Detectors can drift as new generators improve."
)