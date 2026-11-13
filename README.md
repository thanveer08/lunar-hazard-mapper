# 🌑 Lunar Boulder Hazard Mapper

**SW-1 · StarForge**

A web app that finds boulders in lunar orbital images, measures them, and
colour-codes a grid of cells by how hazardous they look for a landing.

> ⚠️ **Research prototype.** It helps prioritise areas for *human review*.
> It is **not** a certified landing-site safety tool. "Clear" means "no boulder
> was detected", not "safe".

---

## What it does

1. You upload a lunar image (JPG, PNG or TIFF, 8-bit or 16-bit).
2. You enter the image's pixel scale (metres per pixel).
3. A YOLOv8 segmentation model detects boulders. Large images are cut into
   640 px tiles so small boulders are not lost.
4. Each boulder's size is estimated in metres (longest side of its box × m/px).
5. The image is divided into square cells (default 5 m × 5 m) and every cell
   with boulders is classified:

| Status | Meaning |
|---|---|
| 🟢 **Clear** | Nothing detected in the cell (not a guarantee) |
| 🟡 **Caution** | Contains hazard-size boulders, but below the density limit |
| 🔵 **Review** | Borderline detections: low confidence, or size within measurement error of the threshold. Needs a human look |
| 🔴 **Unsafe** | Hazard-boulder density ≥ limit, **or** any boulder ≥ the critical size touches the cell |

Conservative mode can raise a cell to *Review* but **never** lowers *Unsafe*.

You can download the annotated hazard map (PNG), the flagged cells (CSV) and
the full detection list (CSV).

## Run it

**Locally**

```bash
pip install -r requirements.txt
streamlit run app.py
```

**On Streamlit Community Cloud:** point it at this repo and set the main file
to `app.py`. `best.pt` must stay next to `app.py`.

To use a different model, set the environment variable `BOULDER_MODEL_PATH`.
(There is deliberately no "model path" text box: loading an arbitrary `.pt`
file can run arbitrary code.)

## Settings

| Setting | What it does |
|---|---|
| Meters per pixel | **Most important input.** Every size depends on it. Read it from the image metadata |
| Hazard size threshold | Boulders at least this big count as hazards |
| Critical boulder size | One boulder this big makes every cell it touches Unsafe |
| Density threshold | Hazard boulders per m² that make a cell Unsafe (0.12 ≈ 3 per 5×5 m cell) |
| Grid cell size | Cell side length. Rounded to a whole number of pixels; the app shows the real size |
| Confidence threshold | Minimum model confidence to keep a detection (changing it is instant) |
| Tiled detection | Recommended for images larger than ~800 px |

## The model

`best.pt` is a **YOLOv8s-seg** model with one class (`boulder`), trained for
50 epochs at 640 px (AdamW, lr 0.001) on a dataset named `boulder2024`.

Validation results stored in the checkpoint:

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Boxes | 0.73 | 0.66 | 0.73 | 0.31 |
| Masks | 0.50 | n/a | 0.43 | 0.13 |

What that means in practice: roughly **1 in 3 boulders is missed**, and about
1 in 4 detections is a false alarm. Box tightness (mAP50-95) is modest, so size
estimates carry real error. The app uses only the boxes, not the masks.

## Known limitations

- Sizes are **lateral** (width), not height. Height, which matters most for
  landing, could be estimated from shadow length and sun angle (not done).
- Boxes may include the boulder's shadow, overestimating size.
- Slopes, craters and surface roughness are not analysed.
- The ±2 pixel uncertainty is a fixed assumption, not a measured value.
- No georeferencing: results are in image pixels, not lunar coordinates.
- Boulders near the hazard threshold are unreliable when the threshold is
  under ~3 pixels (the app warns you).
- The image's pixel scale is typed in by hand.
- The model's training data source, image scale and validation split are not
  documented in this repo.

## Ideas for improvement

- Retrain longer (100–150 epochs) and without rotation/flip/mixup augmentation,
  which distort shadow geometry. Report recall per boulder size.
- Estimate boulder height from shadows.
- Read pixel scale and coordinates from GeoTIFF / PDS metadata.
- Add slope (DEM) and crater layers for a complete landing-hazard map.
- Evaluate on lunar regions not seen in training.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `best.pt` | Trained YOLOv8s-seg weights |
| `requirements.txt` | Pinned Python dependencies |
| `packages.txt` | System packages for Streamlit Cloud (none needed) |

## License note

The app uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
licensed **AGPL-3.0**. If you deploy this app publicly, AGPL obligations
(publishing your source) apply. Add a `LICENSE` file that matches your intent.
