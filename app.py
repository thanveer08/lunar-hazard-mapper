import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import os

st.set_page_config(page_title="Lunar Boulder Hazard Mapper", page_icon="🌑", layout="wide")

@st.cache_resource
def load_model(path):
    return YOLO(path)

def pixel_to_meters(pixel_width, resolution):
    size_m = pixel_width * resolution
    uncertainty_m = 2.0 * resolution
    return size_m, uncertainty_m

def draw_detections(image_rgb, detections):
    out = image_rgb.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf, size = det["conf"], det["size_m"]
        color = (0, 255, 0) if size < 0.5 else (255, 165, 0) if size < 1.0 else (255, 0, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{size:.2f}m | {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out

def build_hazard_grid(image_shape, detections, resolution, grid_size_m, size_thr, density_thr, conservative, conf_thr):
    h, w = image_shape[:2]
    grid_px = max(10, int(grid_size_m / resolution))
    cols, rows = int(np.ceil(w / grid_px)), int(np.ceil(h / grid_px))
    grid = [[[] for _ in range(cols)] for _ in range(rows)]
    grid_all = [[[] for _ in range(cols)] for _ in range(rows)]
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        c, r = min(cols - 1, cx // grid_px), min(rows - 1, cy // grid_px)
        grid_all[r][c].append(det)
        if det["size_m"] >= size_thr:
            grid[r][c].append(det)
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    grid_info = []
    for r in range(rows):
        for c in range(cols):
            hazards, all_det = grid[r][c], grid_all[r][c]
            status, color = "SAFE", (0, 200, 0, 0)
            if len(hazards) >= density_thr:
                status, color = "UNSAFE", (255, 0, 0, 90)
            elif len(hazards) > 0:
                status, color = "CAUTION", (255, 200, 0, 90)
            if conservative and all_det:
                if any(d["conf"] < conf_thr + 0.2 or abs(d["size_m"] - size_thr) < 0.15 for d in all_det):
                    status, color = "REVIEW", (0, 100, 255, 110)
            y0, y1_ = r * grid_px, min(h, (r + 1) * grid_px)
            x0, x1_ = c * grid_px, min(w, (c + 1) * grid_px)
            overlay[y0:y1_, x0:x1_] = color
            grid_info.append({"cell_id": f"R{r}C{c}", "status": status, "n_hazard": len(hazards),
                              "n_total": len(all_det), "max_size_m": max((d["size_m"] for d in all_det), default=0.0)})
    return overlay, grid_info

def composite(image_rgb, overlay_rgba):
    alpha = overlay_rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = overlay_rgba[..., :3].astype(np.float32)
    return (image_rgb.astype(np.float32) * (1 - alpha) + rgb * alpha).astype(np.uint8)

def run_inference(model, image_rgb, resolution, conf_thr):
    results = model.predict(source=image_rgb, conf=conf_thr, verbose=False)
    detections = []
    r = results[0]
    if r.boxes is not None:
        for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box.tolist()
            max_px = max(x2 - x1, y2 - y1)
            size_m, unc_m = pixel_to_meters(max_px, resolution)
            detections.append({"bbox": (x1, y1, x2, y2), "conf": float(conf), "size_m": size_m, "uncertainty_m": unc_m})
    return detections

st.sidebar.header("⚙️ Configuration")
model_path = st.sidebar.text_input("Model path", value="best.pt")
if not os.path.exists(model_path):
    st.sidebar.warning(f"Model not found at '{model_path}'. Using demo model.")
    model_path = "yolov8s-seg.pt"

st.sidebar.subheader("🛰️ Image Resolution")
resolution = st.sidebar.number_input("Meters per pixel", 0.01, 5.0, 0.5, 0.01, help="LROC NAC ≈ 0.5 · Chandrayaan-2 OHRC ≈ 0.25")

st.sidebar.subheader("🚨 Hazard Thresholds")
size_thr = st.sidebar.slider("Hazard size threshold (m)", 0.1, 2.0, 0.5, 0.05)
density_thr = st.sidebar.slider("Density threshold (per cell)", 1, 10, 3, 1)
grid_size_m = st.sidebar.slider("Grid cell size (m)", 1.0, 20.0, 5.0, 0.5)

st.sidebar.subheader("🧠 Conservative Mode")
conservative = st.sidebar.toggle("Enable conservative mode", value=True, help="Flags borderline/low-confidence detections for review.")
conf_thr = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

st.sidebar.subheader("🖼️ Input")
uploaded = st.sidebar.file_uploader("Upload lunar image", type=["jpg", "jpeg", "png", "tif", "tiff"])

st.title("🌑 Lunar Boulder Hazard Mapper")
st.caption("SW-1 · StarForge · Detection + hazard classification with measurement uncertainty")

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    image_rgb = np.array(pil_img)
    with st.spinner("Running detection..."):
        detections = run_inference(model, image_rgb, resolution, conf_thr)
    n_hazards = sum(1 for d in detections if d["size_m"] >= size_thr)
    max_size = max((d["size_m"] for d in detections), default=0.0)
    max_unc = max((d["uncertainty_m"] for d in detections), default=0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Boulders", len(detections))
    c2.metric("Hazard-size", n_hazards)
    c3.metric("Largest", f"{max_size:.2f} m")
    c4.metric("Uncertainty", f"±{max_unc:.2f} m")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original")
        st.image(image_rgb, use_container_width=True)
    with col2:
        st.subheader("🎯 Detections")
        st.image(draw_detections(image_rgb, detections), use_container_width=True)
    st.divider()
    st.subheader("🗺️ Hazard Map")
    overlay, grid_info = build_hazard_grid(image_rgb.shape, detections, resolution, grid_size_m, size_thr, density_thr, conservative, conf_thr)
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
        sel = st.selectbox("Select cell to see reasoning", sorted(cell_map.keys()))
        gi = cell_map[sel]
        a1, a2, a3 = st.columns(3)
        a1.markdown(f"**Status:** {gi['status']}")
        a2.markdown(f"**Boulders:** {gi['n_total']}")
        a3.markdown(f"**Hazard-size:** {gi['n_hazard']}")
    st.divider()
    st.subheader("📋 Detection Details")
    if detections:
        df = pd.DataFrame([{
            "ID": i + 1, "X": int((d["bbox"][0] + d["bbox"][2]) / 2), "Y": int((d["bbox"][1] + d["bbox"][3]) / 2),
            "Size (m)": round(d["size_m"], 3), "± (m)": round(d["uncertainty_m"], 3),
            "Conf": round(d["conf"], 3), "Hazard": "YES" if d["size_m"] >= size_thr else "no",
        } for i, d in enumerate(detections)])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "detections.csv", "text/csv")
else:
    st.info("👈 Upload an image from the sidebar to begin.")
