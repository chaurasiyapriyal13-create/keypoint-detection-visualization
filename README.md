# Keypoint Detection and Visualization System

Detects and visualizes "important points" in an image using three classic
computer-vision algorithms:

| Algorithm | What it finds | Descriptor? | Invariance |
|---|---|---|---|
| **Harris Corner Detection** | Corner-like intensity changes (`cornerHarris`) | No | None (not scale/rotation invariant) |
| **ORB** (Oriented FAST + Rotated BRIEF) | FAST keypoints + orientation | Yes (binary, 32-byte) | Rotation, some scale (pyramid) |
| **SIFT** (Scale-Invariant Feature Transform) | Scale-space extrema (DoG) | Yes (float, 128-dim) | Scale + rotation |

## Files

```
keypoint_system/
├── detectors.py          # Harris / ORB / SIFT detection logic (uniform DetectionResult output)
├── visualize.py           # Drawing: annotated images, heatmap, side-by-side panel, bar chart
├── make_sample_image.py   # Generates a synthetic test image (checkerboard, shapes, text, flat regions)
├── main.py                # CLI entry point
├── sample_images/         # Generated / user-supplied input images
└── output/                # Results (created on run)
```

## Usage

```bash
# Run on the built-in synthetic demo image
python3 main.py

# Run on your own image
python3 main.py --image path/to/photo.jpg

# Custom output directory / max processing size
python3 main.py --image photo.jpg --out-dir results --max-dim 1200
```

### Output files (written to `output/` by default)

- `harris_keypoints.png` — Harris corners drawn as red dots
- `harris_heatmap.png` — Harris response strength as a JET colormap overlay
- `orb_keypoints.png` — ORB keypoints (circle size = scale, line = orientation)
- `sift_keypoints.png` — SIFT keypoints (circle size = scale, line = orientation)
- `comparison_panel.png` — all three side by side
- `summary_chart.png` — bar charts comparing keypoint counts and runtime
- `summary.txt` — plain-text numeric summary (counts, runtime, descriptor shape)

## Using it as a library

```python
import cv2
from detectors import detect_all
from visualize import annotate, side_by_side

image = cv2.imread("photo.jpg")
results = detect_all(image)          # {"Harris": DetectionResult, "ORB": ..., "SIFT": ...}

cv2.imwrite("orb_out.png", annotate(image, results["ORB"]))
cv2.imwrite("panel.png", side_by_side(image, results))

print(results["SIFT"].num_keypoints, "SIFT keypoints found")
```

Each detector can also be tuned individually and called directly:

```python
from detectors import detect_harris, detect_orb, detect_sift

harris = detect_harris(image, block_size=2, ksize=3, k=0.04, threshold_ratio=0.01)
orb    = detect_orb(image, n_features=1000)
sift   = detect_sift(image, contrast_threshold=0.03)
```

## Notes on the algorithms

- **Harris** operates on raw pixel intensity gradients — it has no notion of
  scale, so it works best for finding sharp, well-defined corners (e.g. checkerboards,
  building edges, straight-line intersections). This implementation converts the raw
  response map into a keypoint list via thresholding + local non-max suppression so
  it can be visualized/compared using the same interface as ORB and SIFT.
- **ORB** is fast (binary descriptors, integer FAST detector) and free to use commercially —
  a good default for real-time applications (e.g. SLAM, AR tracking).
- **SIFT** builds a scale-space pyramid of Difference-of-Gaussians and is the most
  robust to scale/rotation/illumination changes, at the cost of being the slowest
  of the three (patent expired in 2020, now free to use in OpenCV).

## Extending

- Add feature **matching** between two images by using `cv2.BFMatcher` (Hamming
  distance for ORB, L2 for SIFT) on the `descriptors` field of `DetectionResult`.
- Swap in other OpenCV detectors (FAST, BRISK, AKAZE, KAZE) by following the same
  `DetectionResult` pattern in `detectors.py`.
