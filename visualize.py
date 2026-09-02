"""
visualize.py
------------
Drawing / visualization utilities for the Keypoint Detection and
Visualization System.

Produces:
  - Per-detector annotated images (keypoints drawn with size/orientation)
  - A Harris "response heatmap" overlay
  - A side-by-side comparison panel (Harris | ORB | SIFT)
  - A bar-chart summary of keypoint counts and runtime
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detectors import DetectionResult


def draw_harris(image: np.ndarray, result: DetectionResult, color=(0, 0, 255)) -> np.ndarray:
    """Draw Harris corners as small filled circles on a copy of the image."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for kp in result.keypoints:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        cv2.circle(out, (x, y), 3, color, -1, lineType=cv2.LINE_AA)
    return out


def draw_harris_heatmap(image: np.ndarray, result: DetectionResult, alpha: float = 0.55) -> np.ndarray:
    """Overlay the raw Harris response map as a heatmap on the image."""
    response = result.extra["response_map"]
    norm = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    base = image.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(heatmap, alpha, base, 1 - alpha, 0)


def draw_rich_keypoints(image: np.ndarray, result: DetectionResult, color=(0, 255, 0)) -> np.ndarray:
    """Draw ORB/SIFT keypoints with circle size = scale and a line = orientation."""
    base = image.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    return cv2.drawKeypoints(
        base, result.keypoints, None, color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def annotate(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Dispatch to the right drawing function based on detector name, then
    stamp a small text label with the keypoint count and runtime."""
    if result.name == "Harris":
        out = draw_harris(image, result)
    else:
        out = draw_rich_keypoints(image, result)

    label = f"{result.name}: {result.num_keypoints} kpts | {result.runtime_ms:.1f} ms"
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def side_by_side(image: np.ndarray, results: dict, order=("Harris", "ORB", "SIFT")) -> np.ndarray:
    """Build a single horizontal comparison panel from a {name: DetectionResult} dict."""
    panels = [annotate(image, results[name]) for name in order if name in results]
    h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != h:
            scale = h / p.shape[0]
            p = cv2.resize(p, (int(p.shape[1] * scale), h))
        resized.append(p)
    return np.hstack(resized)


def summary_chart(results: dict, save_path: str) -> None:
    """Bar chart comparing keypoint counts and runtime across detectors."""
    names = list(results.keys())
    counts = [results[n].num_keypoints for n in names]
    times = [results[n].runtime_ms for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(names, counts, color=["#e74c3c", "#3498db", "#2ecc71"])
    axes[0].set_title("Keypoints Detected")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts):
        axes[0].text(i, v, str(v), ha="center", va="bottom")

    axes[1].bar(names, times, color=["#e74c3c", "#3498db", "#2ecc71"])
    axes[1].set_title("Detection Runtime")
    axes[1].set_ylabel("Milliseconds")
    for i, v in enumerate(times):
        axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom")

    fig.suptitle("Keypoint Detector Comparison")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
