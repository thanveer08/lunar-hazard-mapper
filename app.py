"""
Lunar Boulder Hazard Mapper  (SW-1 · StarForge)

Detects boulders in lunar orbital images with a YOLOv8-seg model, converts the
detections to metres, and classifies a grid of cells as CLEAR / CAUTION /
REVIEW / UNSAFE.

IMPORTANT: this is a research prototype. The model misses roughly a third of
boulders (validation recall ~0.66), so "CLEAR" means "nothing was detected",
NOT "safe to land". See README.md for limitations.
"""

import hashlib
import io
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Lunar Boulder Hazard Mapper", page_icon="🌑", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
# Fixed path on purpose: a free-text model path would let any visitor make the
# server load an arbitrary .pt file (these files can execute code when loaded).
# To use a different model, set the BOULDER_MODEL_PATH environment variable.
MODEL_PATH = os.environ.get("BOULDER_MODEL_PATH", str(APP_DIR / "best.pt"))

ALLOWED_CLASSES = ("boulder", "rock", "stone")

# Status levels, lowest to highest. A status can only ever be escalated.
PRIORITY = {"CLEAR": 0, "CAUTION": 1, "REVIEW": 2, "UNSAFE": 3}
STATUS_STYLE = {
    "CLEAR": {"rgba": (0, 200, 0, 40), "emoji": "🟢", "text": "Clear (nothing detected)"},
    "CAUTION": {"rgba": (255, 200, 0, 90), "emoji": "🟡", "text": "Caution"},
    "REVIEW": {"rgba": (0, 100, 255, 110), "emoji": "🔵", "text": "Needs human review"},
    "UNSAFE": {"rgba": (255, 0, 0, 90), "emoji": "🔴", "text": "Unsafe"},
}

# Tiled inference (the model was trained at 640 px; big images are cut into
# 640 px tiles instead of being shrunk, which would erase small boulders).
TILE = 640
TILE_OVERLAP = 0.20
TILE_BATCH = 8
MAX_TILES = 400
EDGE_MARGIN_PX = 3          # boxes touching an inner tile border are dropped
CONF_FLOOR = 0.05           # run the model once at this floor, filter later

UNCERTAINTY_PX = 2.0        # +/- 2 pixels of size error per detection
CONF_MARGIN = 0.20          # confidence within this of the threshold = borderline
MAX_DISPLAY_PX = 2000       # on-screen images are downscaled to this
MIN_GRID_PX = 2
MIN_AREA_FRACTION = 0.25    # tiny edge cells never get less than 25% of nominal area


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
# Image loading (8-bit, 16-bit, float, grayscale, RGBA, BGR all handled)
# ---------------------------------------------------------------------------
def _to_uint8(arr):
    """Convert any numeric image to uint8 with a percentile contrast stretch."""
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    sample = a[::4, ::4] if a.ndim >= 2 and min(a.shape[:2]) > 8 else a
    lo, hi = np.percentile(sample, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(sample.min()), float(sample.max())
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


@st.cache_resource(max_entries=2, show_spinner=False)
def load_image(file_hash, file_bytes):
    """Decode uploaded bytes to an RGB uint8 array. (file_hash is the cache key.)"""
    buf = np.frombuffer(file_bytes, np.uint8)
    arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if arr is None:
        return np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"))

    arr = _to_uint8(arr)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 2:                      # gray + alpha
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Detection (pixel space)
# ---------------------------------------------------------------------------
def _class_ok(cls_name):
    n = str(cls_name).lower()
    return any(a in n for a in ALLOWED_CLASSES)


def tile_origins(length, tile=TILE, overlap=TILE_OVERLAP):
    """Start offsets of overlapping tiles that cover `length` pixels."""
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, length - tile, step)) + [length - tile]
    return sorted(set(starts))


def count_tiles(shape, use_tiles):
    h, w = shape[:2]
    if not use_tiles or max(h, w) <= TILE * 1.25:
        return 1
    return len(tile_origins(w)) * len(tile_origins(h))


def nms(boxes, scores, iou_thr=0.5, ios_thr=0.8):
    """Greedy NMS that also removes boxes mostly contained in a stronger box."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    areas = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if rest.size == 0:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0)
        ios = np.where(np.minimum(areas[i], areas[rest]) > 0,
                       inter / np.minimum(areas[i], areas[rest]), 0)
        order = rest[(iou < iou_thr) & (ios < ios_thr)]
    return keep


def _predict_batch(model, crops):
    """Run the model on a list of RGB arrays; return one list of dets per crop."""
    # PIL sources are treated as RGB by Ultralytics (numpy arrays would be BGR).
    results = model.predict(
        source=[Image.fromarray(np.ascontiguousarray(c)) for c in crops],
        conf=CONF_FLOOR, imgsz=TILE, verbose=False,
    )
    out = []
    for r in results:
        dets = []
        if r.boxes is not None and len(r.boxes):
            names = r.names if getattr(r, "names", None) else {}
            for box, conf, cls in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
            ):
                dets.append({
                    "bbox": tuple(float(v) for v in box),
                    "conf": float(conf),
                    "class": str(names.get(int(cls), int(cls))),
                })
        out.append(dets)
    return out


def detect_pixels(model, image_rgb, use_tiles):
    """Detect boulders; boxes are in ORIGINAL image pixels, no physical units yet."""
    h, w = image_rgb.shape[:2]
    if count_tiles(image_rgb.shape, use_tiles) == 1:
        return [d for d in _predict_batch(model, [image_rgb])[0] if _class_ok(d["class"])]

    xs, ys = tile_origins(w), tile_origins(h)
    tiles = [(x, y) for y in ys for x in xs]
    if len(tiles) > MAX_TILES:
        raise ValueError(
            f"Image is too large for the free server ({len(tiles)} tiles, limit {MAX_TILES}). "
            "Please crop it to the area of interest."
        )

    merged = []
    for i in range(0, len(tiles), TILE_BATCH):
        batch = tiles[i:i + TILE_BATCH]
        crops = [image_rgb[y:y + TILE, x:x + TILE] for x, y in batch]
        for (x0, y0), crop, dets in zip(batch, crops, _predict_batch(model, crops)):
            th, tw = crop.shape[:2]
            for d in dets:
                if not _class_ok(d["class"]):
                    continue
                bx1, by1, bx2, by2 = d["bbox"]
                # Drop boxes cut off by an INNER tile border: the overlapping
                # neighbour tile sees that boulder whole.
                if ((bx1 <= EDGE_MARGIN_PX and x0 > 0)
                        or (bx2 >= tw - EDGE_MARGIN_PX and x0 + tw < w)
                        or (by1 <= EDGE_MARGIN_PX and y0 > 0)
                        or (by2 >= th - EDGE_MARGIN_PX and y0 + th < h)):
                    continue
                merged.append({**d, "bbox": (bx1 + x0, by1 + y0, bx2 + x0, by2 + y0)})

    keep = nms([d["bbox"] for d in merged], [d["conf"] for d in merged])
    return [merged[i] for i in keep]


@st.cache_data(show_spinner=False, max_entries=6)
def cached_detect(file_hash, model_id, use_tiles, _model, _image):
    """Heavy step, cached: moving any slider does NOT rerun the model."""
    return detect_pixels(_model, _image, use_tiles)


def to_physical(dets_px, resolution):
    """Attach sizes in metres. Size = longest side of the bounding box."""
    out = []
    for d in dets_px:
        x1, y1, x2, y2 = d["bbox"]
        out.append({
            **d,
            "size_m": max(x2 - x1, y2 - y1) * resolution,
            "uncertainty_m": UNCERTAINTY_PX * resolution,
        })
    return out


# ---------------------------------------------------------------------------
# Hazard classification
# ---------------------------------------------------------------------------
def escalate(current, new):
    """Return the more severe of two statuses (never downgrades)."""
    return new if PRIORITY[new] > PRIORITY[current] else current


def effective_grid_px(grid_size_m, resolution):
    return max(MIN_GRID_PX, int(round(grid_size_m / resolution)))


def classify_cell(all_det, critical_size, area_m2, p):
    """Return (status, reasons list, n_hazard, density) for one cell."""
    hazards = [d for d in all_det if d["size_m"] >= p["size_thr"]]
    density = len(hazards) / area_m2
    status, reasons = "CLEAR", []

    if critical_size is not None:
        status = escalate(status, "UNSAFE")
        reasons.append(
            f"A boulder of {critical_size:.2f} m (≥ critical {p['critical_thr']:.2f} m) touches this cell."
        )
    if density >= p["density_thr"]:
        status = escalate(status, "UNSAFE")
        reasons.append(
            f"Hazard-boulder density {density:.3f}/m² ≥ limit {p['density_thr']:.3f}/m²."
        )
    elif hazards:
        status = escalate(status, "CAUTION")
        reasons.append(
            f"{len(hazards)} hazard-size boulder(s) (≥ {p['size_thr']:.2f} m), density below limit."
        )

    if p["conservative"]:
        borderline = [
            d for d in all_det
            if d["conf"] < p["conf_thr"] + CONF_MARGIN
            or abs(d["size_m"] - p["size_thr"]) <= d["uncertainty_m"]
        ]
        if borderline:
            # escalate() can raise CLEAR/CAUTION to REVIEW but never lowers UNSAFE.
            status = escalate(status, "REVIEW")
            reasons.append(
                f"{len(borderline)} detection(s) are borderline (low confidence, or size within "
                f"measurement error of the {p['size_thr']:.2f} m threshold)."
            )

    if not reasons:
        reasons.append("Only small, confident detections; nothing near the hazard thresholds.")
    return status, reasons, len(hazards), density


def build_hazard_grid(image_shape, detections, resolution, grid_size_m, params, scale=1.0):
    """
    Returns (overlay_rgba at display scale, grid_info list, summary dict).
    Only cells that contain detections (or touch a critical boulder) get entries;
    every other cell is CLEAR.
    """
    h, w = image_shape[:2]
    grid_px = effective_grid_px(grid_size_m, resolution)
    cell_m = grid_px * resolution
    cols, rows = int(np.ceil(w / grid_px)), int(np.ceil(h / grid_px))

    cell_dets = {}
    critical_cells = {}

    def clamp(v, hi):
        return max(0, min(hi, int(v)))

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        r = clamp(((y1 + y2) / 2) // grid_px, rows - 1)
        c = clamp(((x1 + x2) / 2) // grid_px, cols - 1)
        cell_dets.setdefault((r, c), []).append(det)
        if det["size_m"] >= params["critical_thr"]:
            # Footprint rule: a big boulder endangers EVERY cell it touches.
            for rr in range(clamp(y1 // grid_px, rows - 1), clamp(y2 // grid_px, rows - 1) + 1):
                for cc in range(clamp(x1 // grid_px, cols - 1), clamp(x2 // grid_px, cols - 1) + 1):
                    critical_cells[(rr, cc)] = max(critical_cells.get((rr, cc), 0.0), det["size_m"])

    hd, wd = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    overlay = np.zeros((hd, wd, 4), dtype=np.uint8)
    overlay[:] = STATUS_STYLE["CLEAR"]["rgba"]

    grid_info = []
    for (r, c) in sorted(set(cell_dets) | set(critical_cells)):
        all_det = cell_dets.get((r, c), [])
        y0, y1 = r * grid_px, min(h, (r + 1) * grid_px)
        x0, x1 = c * grid_px, min(w, (c + 1) * grid_px)
        # True area of this cell (edge cells are partial), with a sane floor.
        area_m2 = max((x1 - x0) * (y1 - y0) * resolution ** 2, MIN_AREA_FRACTION * cell_m ** 2)

        status, reasons, n_haz, density = classify_cell(
            all_det, critical_cells.get((r, c)), area_m2, params
        )
        overlay[int(round(y0 * scale)):int(round(y1 * scale)),
                int(round(x0 * scale)):int(round(x1 * scale))] = STATUS_STYLE[status]["rgba"]
        grid_info.append({
            "cell_id": f"R{r}C{c}",
            "status": status,
            "n_hazard": n_haz,
            "n_total": len(all_det),
            "density": round(density, 4),
            "max_size_m": round(max((d["size_m"] for d in all_det), default=0.0), 3),
            "x_px": x0, "y_px": y0,
            "reason": " ".join(reasons),
        })

    counts = {k: 0 for k in PRIORITY}
    for gi in grid_info:
        counts[gi["status"]] += 1
    counts["CLEAR"] = rows * cols - len(grid_info) + counts["CLEAR"]
    summary = {"rows": rows, "cols": cols, "cell_m": cell_m, "grid_px": grid_px, "counts": counts}
    return overlay, grid_info, summary


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def display_scale(shape):
    return min(1.0, MAX_DISPLAY_PX / max(shape[:2]))


def draw_detections(disp_rgb, detections, scale, size_thr, critical_thr):
    out = disp_rgb.copy()
    label = len(detections) <= 250   # skip text on very crowded images
    for det in detections:
        x1, y1, x2, y2 = (int(round(v * scale)) for v in det["bbox"])
        size = det["size_m"]
        if size < size_thr:
            color = (0, 255, 0)          # below hazard size
        elif size < critical_thr:
            color = (255, 165, 0)        # hazard size
        else:
            color = (255, 0, 0)          # critical size
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if label:
            text = f"{size:.2f}m | {det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def composite(image_rgb, overlay_rgba, band=256):
    """Alpha-blend in row bands so huge images do not need huge float buffers."""
    out = np.empty_like(image_rgb)
    for y in range(0, image_rgb.shape[0], band):
        img = image_rgb[y:y + band].astype(np.float32)
        ov = overlay_rgba[y:y + band]
        alpha = ov[..., 3:4].astype(np.float32) / 255.0
        out[y:y + band] = (img * (1 - alpha) + ov[..., :3].astype(np.float32) * alpha).astype(np.uint8)
    return out


def cell_sort_key(cid):
    r_str, c_str = cid[1:].split("C")
    return int(r_str), int(c_str)


def png_bytes(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("🛰️ Image Resolution")
resolution = st.sidebar.number_input(
    "Meters per pixel", 0.01, 5.0, 0.5, 0.01,
    help="LROC NAC ≈ 0.5 · Chandrayaan-2 OHRC ≈ 0.25 · Apollo ≈ 0.1–0.5 (varies). "
         "Every size in this app depends on this number: check your image metadata.",
)

st.sidebar.subheader("🚨 Hazard Thresholds")
size_thr = st.sidebar.slider("Hazard size threshold (m)", 0.1, 5.0, 0.5, 0.05)
critical_thr = st.sidebar.slider(
    "Critical boulder size (m)", 0.5, 10.0, 2.0, 0.1,
    help="A single boulder at least this large makes every cell it touches UNSAFE, "
         "regardless of density. Set it to the largest obstacle your lander can tolerate.",
)
density_thr = st.sidebar.slider(
    "Density threshold (boulders / m²)", 0.01, 2.0, 0.12, 0.01,
    help="Default 0.12 ≈ 3 hazard boulders per 5m × 5m cell",
)
grid_size_m = st.sidebar.slider("Grid cell size (m)", 1.0, 20.0, 5.0, 0.5)

st.sidebar.subheader("🧠 Conservative Mode")
conservative = st.sidebar.toggle(
    "Enable conservative mode", value=True,
    help="Flags borderline / low-confidence detections as REVIEW. Never downgrades UNSAFE cells.",
)
conf_thr = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

st.sidebar.subheader("🔬 Detection")
use_tiles = st.sidebar.toggle(
    "Tiled detection for large images", value=True,
    help="Cuts big images into 640 px tiles so small boulders are not lost when the "
         "image is shrunk. Slower, but much more accurate on large images.",
)

st.sidebar.subheader("🖼️ Input")
uploaded = st.sidebar.file_uploader(
    "Upload lunar image", type=["jpg", "jpeg", "png", "tif", "tiff"]
)

st.sidebar.divider()
st.sidebar.caption(
    "Model: YOLOv8s-seg, 1 class (boulder). Validation: precision 0.73 · recall 0.66 · "
    "mAP50 0.73 · mAP50-95 0.31. About 1 in 3 boulders is missed."
)

# ---------------------------------------------------------------------------
# Title & model
# ---------------------------------------------------------------------------
st.title("🌑 Lunar Boulder Hazard Mapper")
st.caption("SW-1 · StarForge · Detection + hazard classification with measurement uncertainty")

try:
    model, is_demo = load_model(MODEL_PATH)
except Exception as e:  # noqa: BLE001
    st.error(
        f"Failed to load the model: {e}\n\n"
        "Make sure `best.pt` is in the same folder as `app.py`."
    )
    st.stop()

if is_demo:
    st.warning(
        f"⚠️ **Demo mode**: no model found at `{MODEL_PATH}`. "
        "Using the YOLO-World open-vocabulary fallback (`boulder`, `rock`, `stone`). "
        "Detections are illustrative; do not trust the size numbers for real planning."
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if not uploaded:
    st.info("👈 Upload an image from the sidebar to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()
image_rgb = load_image(file_hash, file_bytes)
H, W = image_rgb.shape[:2]

n_tiles = count_tiles(image_rgb.shape, use_tiles)
try:
    with st.spinner(f"Running detection ({n_tiles} tile{'s' if n_tiles != 1 else ''})..."):
        all_px = cached_detect(file_hash, MODEL_PATH, use_tiles, model, image_rgb)
except ValueError as e:
    st.error(str(e))
    st.stop()

dets_px = [d for d in all_px if d["conf"] >= conf_thr]
detections = to_physical(dets_px, resolution)

n_hazards = sum(1 for d in detections if d["size_m"] >= size_thr)
max_size = max((d["size_m"] for d in detections), default=0.0)
unc = UNCERTAINTY_PX * resolution

c1, c2, c3, c4 = st.columns(4)
c1.metric("Boulders", len(detections))
c2.metric("Hazard-size", n_hazards)
c3.metric("Largest", f"{max_size:.2f} m")
c4.metric("± per detection", f"{unc:.2f} m",
          help=f"±{UNCERTAINTY_PX:g} pixels of size error: verify m/px against image metadata")

st.caption(
    f"Image: {W}×{H} px ≈ {W * resolution:,.0f} × {H * resolution:,.0f} m at **{resolution} m/px**. "
    "All sizes assume this scale; check the image's real pixel scale before trusting them."
)

if size_thr < 3 * resolution:
    st.warning(
        f"⚠️ The hazard size threshold ({size_thr:.2f} m) is under 3 pixels at {resolution} m/px "
        f"(3 px = {3 * resolution:.2f} m). Boulders that small are barely visible, so results near "
        "this threshold are unreliable. Many detections will be flagged for review."
    )
st.warning(
    "**'Clear' ≠ 'safe'.** The model misses about 1 in 3 boulders, and slopes and craters "
    "are not analysed. Use this map to prioritise human review, not to approve a landing site."
)

st.divider()

scale = display_scale(image_rgb.shape)
disp = (cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0 else image_rgb)
if scale < 1.0:
    st.caption(f"Display downscaled to {disp.shape[1]}×{disp.shape[0]} px for speed "
               "(measurements use the full-resolution image).")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 Original")
    st.image(disp, width="stretch")
with col2:
    st.subheader("🎯 Detections")
    det_img = draw_detections(disp, detections, scale, size_thr, critical_thr)
    st.image(det_img, width="stretch")
    st.caption("🟢 below hazard size · 🟠 hazard size · 🔴 critical size")

st.divider()
st.subheader("🗺️ Hazard Map")

params = {
    "size_thr": size_thr, "critical_thr": critical_thr, "density_thr": density_thr,
    "conservative": conservative, "conf_thr": conf_thr,
}
overlay, grid_info, summary = build_hazard_grid(
    image_rgb.shape, detections, resolution, grid_size_m, params, scale=scale
)
hazard_map = composite(disp, overlay)
st.image(hazard_map, width="stretch")

if abs(summary["cell_m"] - grid_size_m) > 0.01:
    st.caption(
        f"Grid cells are {summary['grid_px']} px = **{summary['cell_m']:.2f} m** "
        f"(rounded from the requested {grid_size_m:.1f} m to a whole number of pixels). "
        "Density uses this real cell size."
    )

counts = summary["counts"]
lc = st.columns(4)
for col, key in zip(lc, ("CLEAR", "CAUTION", "UNSAFE", "REVIEW")):
    s = STATUS_STYLE[key]
    col.markdown(f"{s['emoji']} **{s['text']}**  \n{counts[key]:,} cells")

st.download_button("⬇️ Download hazard map (PNG)", png_bytes(hazard_map),
                   "hazard_map.png", "image/png")

st.divider()
st.subheader("🔍 Grid Cell Audit")
cell_map = {gi["cell_id"]: gi for gi in grid_info}
if cell_map:
    sorted_ids = sorted(cell_map.keys(), key=cell_sort_key)
    sel = st.selectbox("Select cell to see reasoning", sorted_ids)
    gi = cell_map[sel]
    a1, a2, a3, a4 = st.columns(4)
    a1.markdown(f"**Status:** {STATUS_STYLE[gi['status']]['emoji']} {gi['status']}")
    a2.markdown(f"**Boulders:** {gi['n_total']}")
    a3.markdown(f"**Hazard-size:** {gi['n_hazard']}")
    a4.markdown(f"**Density:** {gi['density']:.3f} / m²")
    st.info(f"**Why:** {gi['reason']}")

    cells_df = pd.DataFrame(grid_info).rename(columns={
        "cell_id": "Cell", "status": "Status", "n_hazard": "Hazard boulders",
        "n_total": "All boulders", "density": "Density (/m²)",
        "max_size_m": "Largest (m)", "x_px": "X (px)", "y_px": "Y (px)", "reason": "Reason",
    })
    st.download_button(
        "⬇️ Download flagged cells (CSV)", cells_df.to_csv(index=False).encode(),
        "hazard_cells.csv", "text/csv",
        help="Lists every cell that is not plain CLEAR-with-no-detections.",
    )
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
        "± (m)": round(d["uncertainty_m"], 3),
        "Conf": round(d["conf"], 3),
        "Hazard": "YES" if d["size_m"] >= size_thr else "no",
    } for i, d in enumerate(detections)])
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Download detections (CSV)", df.to_csv(index=False).encode(),
        "detections.csv", "text/csv",
    )
else:
    st.info("No detections above the confidence threshold. "
            "Try lowering it, or check that your model's class names include "
            "'boulder', 'rock', or 'stone'.")

with st.expander("ℹ️ How this works & its limits"):
    st.markdown(
        """
- **Size** is the longest side of the detection box × metres-per-pixel. It ignores boulder
  *height* (which matters most for landing) and can include the shadow.
- **± uncertainty** is a fixed ±2 pixels. Boulders whose size is within that band of the
  threshold are flagged **REVIEW** in conservative mode.
- **UNSAFE** if hazard-boulder density ≥ the limit, **or** any boulder ≥ the critical size
  touches the cell. Conservative mode can raise a cell to REVIEW but never lowers UNSAFE.
- **Clear** only means nothing was detected. The model has ~66% recall.
- Not analysed: slopes, craters, illumination angle, georeferencing.
"""
    )
