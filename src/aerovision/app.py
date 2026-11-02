"""Local image-upload demo backed exclusively by ONNX Runtime CPU."""

import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from aerovision.runtime import CpuDetector, annotate, load_config

ROOT = Path(__file__).resolve().parents[2]


@st.cache_resource
def detector():
    return CpuDetector(
        ROOT / "models/aerovision-full-fp32.onnx", load_config(ROOT / "configs/frozen-v1.json")
    )


def main():
    st.set_page_config(page_title="AeroVision Lab", page_icon="◈", layout="wide")
    st.markdown(
        """<style>
    .block-container {max-width:1400px; padding-top:2.4rem;}
    .eyebrow {color:#55D5CE; letter-spacing:.18em; font-size:.76rem; font-weight:700;}
    .subhead {color:#A3B4CB; font-size:1.08rem; max-width:720px; margin-bottom:1.6rem;}
    h1 {letter-spacing:-.045em; font-size:3.1rem !important;}
    [data-testid="stMetricValue"] {font-size:1.65rem;}
    .empty {border:1px dashed #33465F; border-radius:16px; padding:72px 28px;
            text-align:center; color:#A3B4CB; background:#101A2B;}
    .empty b {color:#E8EEF7; font-size:1.4rem; display:block; margin:16px 0 8px;}
    .foot {font-size:.8rem; color:#8293AD; margin-top:18px;}
    </style>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">PERCEPTION / EXPERIMENT / DEPLOYMENT</div>', unsafe_allow_html=True)
    st.title("AeroVision Lab")
    st.markdown(
        '<div class="subhead">Explore what a compact vision model sees from above. '
        "Upload an aerial image and inspect its detections.</div>",
        unsafe_allow_html=True,
    )
    explore, evidence = st.tabs(["Image explorer", "Evidence & limitations"])

    with explore:
        control, canvas = st.columns([1, 2.8], gap="large")
        with control:
            st.subheader("Your image")
            upload = st.file_uploader("Upload JPG or PNG", type=["jpg", "jpeg", "png"])
            confidence = st.slider("Confidence threshold", 0.05, 0.80, 0.30, 0.01)
            st.caption("0.30 is the validation-selected default. Lower values show more predictions.")
            show_labels = st.checkbox("Show labels", value=True)
            st.divider()
            st.markdown("**YOLO11n · frozen v1**")
            st.caption("1280 input · 10 classes · CPU inference")
            ready = (ROOT / "models/aerovision-full-fp32.onnx").exists()
            if ready:
                st.success("Model available")
            else:
                st.warning("Model not installed. Follow the model setup in the README.")
            run = st.button(
                "Detect objects",
                type="primary",
                use_container_width=True,
                disabled=upload is None or not ready,
            )

        with canvas:
            if upload is None:
                st.markdown(
                    '<div class="empty">◈<b>A new perspective</b>'
                    "Add an aerial street or traffic image to explore detections."
                    "<br>Images stay in this local session.</div>",
                    unsafe_allow_html=True,
                )
            else:
                data = upload.getvalue()
                try:
                    source = Image.open(io.BytesIO(data))
                    if source.width * source.height > 20_000_000:
                        st.error("Choose an image below 20 megapixels.")
                        st.stop()
                    image = ImageOps.exif_transpose(source).convert("RGB")
                except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                    st.error("This file could not be read as an image. Try another JPG or PNG.")
                    st.stop()
                key = hashlib.sha256(data).hexdigest() + f"-{confidence:.2f}"
                if run:
                    try:
                        with st.spinner("Inspecting image on CPU…"):
                            result = detector().predict(image, confidence=confidence)
                        st.session_state["last_result"] = (key, result)
                    except (ValueError, OSError, RuntimeError) as error:
                        st.error(f"Inference could not complete: {error}")
                saved = st.session_state.get("last_result")
                if saved and saved[0] == key:
                    result = saved[1]
                    rows = result["detections"]
                    a, b, c = st.columns(3)
                    a.metric("Detections", len(rows))
                    b.metric("Classes found", len({r["class_id"] for r in rows}))
                    c.metric("This CPU inference", f"{result['latency_ms']:.0f} ms")
                    drawn = annotate(image, rows, labels=show_labels)
                    st.image(
                        drawn,
                        caption="Model predictions; boxes are not verified ground truth.",
                        use_container_width=True,
                    )
                    counts = Counter(r["label"] for r in rows)
                    st.caption(
                        " · ".join(f"{name}: {count}" for name, count in counts.most_common())
                        if rows
                        else "No detections above this threshold."
                    )
                    image_bytes = io.BytesIO()
                    drawn.save(image_bytes, format="PNG")
                    left, right = st.columns(2)
                    left.download_button(
                        "Download annotated image",
                        image_bytes.getvalue(),
                        file_name="aerovision-detections.png",
                        mime="image/png",
                    )
                    right.download_button(
                        "Download predictions",
                        json.dumps(result, indent=2),
                        file_name="aerovision-predictions.json",
                        mime="application/json",
                    )
                    with st.expander("Inspect detection values"):
                        st.dataframe(
                            [
                                dict(
                                    label=r["label"],
                                    confidence=round(r["confidence"], 3),
                                    **dict(zip(["x1", "y1", "x2", "y2"], r["box_xyxy"])),
                                )
                                for r in rows
                            ],
                            use_container_width=True,
                        )
                else:
                    st.image(
                        image,
                        caption="Image ready. Select Detect objects to run inference.",
                        use_container_width=True,
                    )

    with evidence:
        st.subheader("A measured improvement on held-out images")
        st.write(
            "One model, two inference resolutions, 1,610 test-dev images. The model was "
            "trained on all 6,471 training images for 60 epochs on a hosted T4."
        )
        a, b, c = st.columns(3)
        a.metric("Held-out AP", "18.61%", "+4.92 pp vs 640")
        b.metric("Small-object AP", "11.30%", "+6.44 pp vs 640")
        c.metric("Recorded T4 median", "17.1 ms")
        st.caption("The recorded T4 measurement is not the CPU runtime shown in Image explorer.")
        st.dataframe(
            [
                {
                    "Configuration": "Whole image · 640",
                    "AP": "13.69%",
                    "Small-object AP": "4.86%",
                    "T4 median": "12.3 ms",
                },
                {
                    "Configuration": "Whole image · 1280",
                    "AP": "18.61%",
                    "Small-object AP": "11.30%",
                    "T4 median": "17.1 ms",
                },
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**Where it struggles**")
        st.write(
            "At confidence 0.30, held-out car recall was 77.4%, pedestrian recall 28.8%, "
            "and bicycle recall 4.2%. Dense crowds, tiny objects and unfamiliar scenes "
            "can produce substantial misses and false positives."
        )
        st.caption(
            "Scores use a custom COCO-style protocol with an ignored-region approximation, "
            "not the official VisDrone challenge evaluator. One training seed was tested. "
            "ONNX parity was checked on ten validation images and additional synthetic shapes; "
            "full test-dev accuracy was measured with PyTorch."
        )
    st.markdown(
        '<div class="foot">AeroVision Lab · Frozen experiment v1 · '
        "ONNX Runtime CPU · Uploads are not written to disk by the app</div>",
        unsafe_allow_html=True,
    )
