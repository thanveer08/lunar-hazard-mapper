import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import os

st.set_page_config(page_title="Lunar Boulder Hazard Mapper", page_icon="🌑", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_CLASSES = ("boulder", "rock", "stone")
PRIORITY = {"SAFE": 0, "CAUTION": 1, "REVIEW": 2, "UNSAFE": 3}
COLORS = {
    "SAFE":    (0, 200, 0, 40),      # faint green tint so the legend matches
    "CAUTION": (255, 200, 0, 90),
    "REVIEW":  (0, 100, 255, 110),
    "UNSAFE":  (255, 0, 0, 90),
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    """Return (model, is_demo). Falls back to YOLO-World open-vocab if path is missing."""
    if os.path.exists(path):
        return YOLO(path), False
    from ultralytics import YOLOWorld
    m = YOLOWorld("yolov8s-world.pt")
    m.set_classes(list(ALLOWED_CLASSES))
    return m, True

# ---------------------------------------------------------------------------
# Image loading (8-bit, 16-bit, grayscale, RGBA, BGR — all handled)
# ---------------------------------------------------------------------------
def load_image(file):
    file_bytes = np.frombuffer(file.getvalue(), np.uint8)
    arr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if arr is None:
        return np.array(Image.open(file).convert("RGB"))
    if arr.dtype == np.uint16:
        arr = (arr / 256.0).clip(0, 255).astype(np.uint8)   # LROC 16-bit
    elif arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _class_ok(cls_name):
    n = cls_name.lower()
    return any(a in n for a in ALLOWED_CLASSES)

def run_inference(model, image_rgb, resolution, conf_thr):
    # PIL source → Ultralytics treats as RGB (no BGR swap)
    results = model.predict(source=Image.fromarray(image_rgb), conf=conf_thr, verbose=False)
    detections = []
    r = results[0]
    if r.boxes is None:
        return detections
    names = r.names if hasattr(r, "names") and r.names else {}
    for box, conf, cls in zip(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.conf.cpu().numpy(),
        r.boxes.cls.cpu().numpy(),
    ):
        cls_name = str(names.get(int(cls), int(cls)))
        if not _class_ok(cls_name):
            continue
        x1, y1, x2, y2 = box.tolist()
        max_px = max(x2 - x1, y2 - y1)
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "conf": float(conf),
            "size_m": max_px * resolution,
            "uncertainty_m": 2.0 * resolution,
            "class": cls_name,
        })
    return detections

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def draw_detections(image_rgb, detections, size_thr):
    out = image_rgb.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        size = det["size_m"]
        if size < size_thr:
            color = (0, 255, 0)
        elif size < 2 * size_thr:
            color = (255, 165, 0)
        else:
            color = (255, 0, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{size:.2f}m | {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out

def build_hazard_grid(image_shape, detections, resolution, grid_size_m,
                      size_thr, density_thr, conservative, conf_thr):
    h, w = image_shape[:2]
    grid_px = max(10, int(round(grid_size_m / resolution)))
    cols, rows = int(np.ceil(w / grid_px)), int(np.ceil(h / grid_px))
    hazards_grid = [[[] for _ in range(cols)] for _ in range(rows)]
    all_grid     = [[[] for _ in range(cols)] for _ in range(rows)]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        c = min(cols - 1, max(0, cx // grid_px))
        r = min(rows - 1, max(0, cy // grid_px))
        all_grid[r][c].append(det)
        if det["size_m"] >= size_thr:
            hazards_grid[r][c].append(det)

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    grid_info = []
    for r in range(rows):
        for c in range(cols):
            hazards, all_det = hazards_grid[r][c], all_grid[r][c]

            y0, y1_ = r * grid_px, min(h, (r + 1) * grid_px)
            x0, x1_ = c * grid_px, min(w, (c + 1) * grid_px)
            cell_area = max(1e-6, (x1_ - x0) * resolution * (y1_ - y0) * resolution)

            # Base classification
            density = len(hazards) / cell_area
            if density >= density_thr:
                status = "UNSAFE"
            elif hazards:
                status = "CAUTION"
            else:
                status = "SAFE"

            # Conservative review: can raise CAUTION/SAFE → REVIEW,
            # but never downgrades UNSAFE.
            if (conservative and all_det and PRIORITY[status] < PRIORITY["UNSAFE"]):
                near_size = any(abs(d["size_m"] - size_thr) < 0.15 for d in all_det)
                low_conf  = any(d["conf"] < conf_thr + 0.2 for d in all_det)
                if near_size or low_conf:
                    status = "REVIEW"

            overlay[y0:y1_, x0:x1_] = COLORS[status]
            grid_info.append({
                "cell_id":    f"R{r}C{c}",
                "status":     status,
                "n_hazard":   len(hazards),
                "n_total":    len(all_det),
                "area_m2":    round(cell_area, 2),
                "density":    round(density, 3),
                "max_size_m": round(max((d["size_m"] for d in all_det), default=0.0), 2),
            })
    return overlay, grid_info

def composite(image_rgb, overlay_rgba):
    alpha = overlay_rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = overlay_rgba[..., :3].astype(np.float32)
    return (image_rgb.astype(np.float32) * (1 - alpha) + rgb * alpha).astype(np.uint8)

def cell_sort_key(cid):
    r_str, c_str = cid[1:].split("C")
    return int(r_str), int(c_str)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")
model_path = st.sidebar.text_input("Model path", value="best.pt")

st.sidebar.subheader("🛰️ Image Resolution")
resolution = st.sidebar.number_input(
    "Meters per pixel", 0.01, 5.0, 0.5, 0.01,
    help="LROC NAC ≈ 0.5 · Chandrayaan-2 OHRC ≈ 0.25 · Apollo ≈ 0.1–0.5 (varies)"
)

st.sidebar.subheader("🚨 Hazard Thresholds")
size_thr = st.sidebar.slider("Hazard size threshold (m)", 0.1, 2.0, 0.5, 0.05)
density_thr = st.sidebar.slider(
    "Density threshold (boulders / m²)", 0.01, 2.0, 0.12, 0.01,
    help="Default 0.12 ≈ 3 hazard boulders per 5m × 5m cell"
)
grid_size_m = st.sidebar.slider("Grid cell size (m)", 1.0, 20.0, 5.0, 0.5)

st.sidebar.subheader("🧠 Conservative Mode")
conservative = st.sidebar.toggle(
    "Enable conservative mode", value=True,
    help="Flags borderline/low-confidence detections for review. Never downgrades UNSAFE cells."
)
conf_thr = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

st.sidebar.subheader("🖼️ Input")
uploaded = st.sidebar.file_uploader(
    "Upload lunar image", type=["jpg", "jpeg", "png", "tif", "tiff"]
)

# ---------------------------------------------------------------------------
# Title & model
# ---------------------------------------------------------------------------
st.title("🌑 Lunar Boulder Hazard Mapper")
st.caption("SW-1 · StarForge · Detection + hazard classification with measurement uncertainty")

try:
    model, is_demo = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

if is_demo:
    st.warning(
        f"⚠️ **Demo mode** — no model found at `{model_path}`. "
        "Using YOLO-World open-vocabulary fallback (`boulder`, `rock`, `stone`). "
        "Detections are illustrative; do not trust the size numbers for real planning."
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if uploaded:
    image_rgb = load_image(uploaded)
    with st.spinner("Running detection..."):
        detections = run_inference(model, image_rgb, resolution, conf_thr)

    n_hazards = sum(1 for d in detections if d["size_m"] >= size_thr)
    max_size  = max((d["size_m"] for d in detections), default=0.0)
    unc       = 2.0 * resolution

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Boulders", len(detections))
    c2.metric("Hazard-size", n_hazards)
    c3.metric("Largest", f"{max_size:.2f} m")
    c4.metric("± per detection", f"{unc:.2f} m",
              help="2-pixel resolution uncertainty — verify m/px against image metadata")
    st.caption(
        f"⚠️ All sizes assume **{resolution} m/px**. "
        "Check the image's real pixel scale before trusting these numbers."
    )
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original")
        st.image(image_rgb, use_container_width=True)
    with col2:
        st.subheader("🎯 Detections")
        st.image(draw_detections(image_rgb, detections, size_thr),
                 use_container_width=True)

    st.divider()
    st.subheader("🗺️ Hazard Map")
    overlay, grid_info = build_hazard_grid(
        image_rgb.shape, detections, resolution, grid_size_m,
        size_thr, density_thr, conservative, conf_thr
    )
    st.image(composite(image_rgb, overlay), use_container_width=True)
    lc = st.columns(4)
    lc[0].markdown("🟢 **SAFE**")
    lc[1].markdown("🟡 **CAUTION**")
    lc[2].markdown("🔴 **UNSAFE**")
    lc[3].markdown("🔵 **REVIEW**")

    st.divider()
    st.subheader("🔍 Grid Cell Audit")
    cell_map = {gi["cell_id"]: gi for gi in grid_info if gi["n_total"] > 0}
    if cell_map:
        sorted_ids = sorted(cell_map.keys(), key=cell_sort_key)
        sel = st.selectbox("Select cell to see reasoning", sorted_ids)
        gi = cell_map[sel]
        a1, a2, a3, a4 = st.columns(4)
        a1.markdown(f"**Status:** {gi['status']}")
        a2.markdown(f"**Boulders:** {gi['n_total']}")
        a3.markdown(f"**Hazard-size:** {gi['n_hazard']}")
        a4.markdown(f"**Density:** {gi['density']:.3f} / m²")
    else:
        st.info("No boulders detected in any grid cell.")

    st.divider()
    st.subheader("📋 Detection Details")
    if detections:
        df = pd.DataFrame([{
            "ID": i + 1,
            "Class": d["class"],
            "X": int((d["bbox"][0] + d["bbox"][2]) / 2),
            "Y": int((d["bbox"][1] + d["bbox"][3]) / 2),
            "Size (m)": round(d["size_m"], 3),
            "± (m)":    round(d["uncertainty_m"], 3),
            "Conf":     round(d["conf"], 3),
            "Hazard":   "YES" if d["size_m"] >= size_thr else "no",
        } for i, d in enumerate(detections)])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False).encode(),
            "detections.csv",
            "text/csv",
        )
    else:
        st.info("No detections above the confidence threshold. "
                "Check that your model's class names include 'boulder', 'rock', or 'stone'.")
else:
    st.info("👈 Upload an image from the sidebar to begin.")
