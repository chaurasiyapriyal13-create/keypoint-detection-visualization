"""
detectors.py
------------
Core detection routines for the Keypoint Detection and Visualization System.

Implements three classic keypoint/feature detectors:
  1. Harris Corner Detection  -> corner "strength" map, no descriptors
  2. ORB (Oriented FAST + Rotated BRIEF) -> keypoints + binary descriptors
  3. SIFT (Scale-Invariant Feature Transform) -> keypoints + float descriptors

Each detector is wrapped in a function that returns a uniform result
dictionary so the visualization / comparison layer can treat them
interchangeably.
"""

from dataclasses import dataclass, field
from typing import Optional
import time

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Uniform container for the output of any detector."""
    name: str
    keypoints: list                      # list[cv2.KeyPoint] (empty for raw Harris)
    descriptors: Optional[np.ndarray]     # None for raw Harris
    num_keypoints: int
    runtime_ms: float
    extra: dict = field(default_factory=dict)  # detector-specific extras (e.g. corner map)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# --------------------------------------------------------------------------
# 1. Harris Corner Detection
# --------------------------------------------------------------------------
def detect_harris(
    image: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold_ratio: float = 0.01,
    nms_window: int = 5,
) -> DetectionResult:
    """
    Classic Harris corner detector (cv2.cornerHarris), converted into a
    keypoint list via thresholding + simple non-maximum suppression so it
    can be visualized/compared the same way as ORB/SIFT.

    Parameters
    ----------
    block_size : neighborhood size considered for corner detection
    ksize      : aperture parameter for the Sobel derivative
    k          : Harris free parameter (typically 0.04 - 0.06)
    threshold_ratio : fraction of the max response used as a keypoint threshold
    nms_window : window size (pixels) for local non-max suppression
    """
    gray = _to_gray(image)
    gray_f = np.float32(gray)

    t0 = time.perf_counter()
    response = cv2.cornerHarris(gray_f, block_size, ksize, k)
    response = cv2.dilate(response, None)  # helps mark corners more visibly

    # Threshold
    thresh = threshold_ratio * response.max()
    corner_mask = response > thresh

    # Simple non-maximum suppression over a local window so we don't get a
    # dense blob of "keypoints" at every corner.
    keypoints = []
    ys, xs = np.where(corner_mask)
    coords = list(zip(ys, xs))
    coords.sort(key=lambda yx: -response[yx[0], yx[1]])  # strongest first

    occupied = np.zeros_like(corner_mask, dtype=bool)
    half = nms_window // 2
    h, w = response.shape
    for y, x in coords:
        if occupied[y, x]:
            continue
        keypoints.append(cv2.KeyPoint(float(x), float(y), size=float(nms_window),
                                       response=float(response[y, x])))
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        occupied[y0:y1, x0:x1] = True

    runtime_ms = (time.perf_counter() - t0) * 1000

    return DetectionResult(
        name="Harris",
        keypoints=keypoints,
        descriptors=None,
        num_keypoints=len(keypoints),
        runtime_ms=runtime_ms,
        extra={"response_map": response, "corner_mask": corner_mask},
    )


# --------------------------------------------------------------------------
# 2. ORB
# --------------------------------------------------------------------------
def detect_orb(
    image: np.ndarray,
    n_features: int = 500,
    scale_factor: float = 1.2,
    n_levels: int = 8,
) -> DetectionResult:
    gray = _to_gray(image)

    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=scale_factor, nlevels=n_levels)

    t0 = time.perf_counter()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    runtime_ms = (time.perf_counter() - t0) * 1000

    return DetectionResult(
        name="ORB",
        keypoints=list(keypoints),
        descriptors=descriptors,
        num_keypoints=len(keypoints),
        runtime_ms=runtime_ms,
    )


# --------------------------------------------------------------------------
# 3. SIFT
# --------------------------------------------------------------------------
def detect_sift(
    image: np.ndarray,
    n_features: int = 0,          # 0 = no limit
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10,
) -> DetectionResult:
    gray = _to_gray(image)

    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
    )

    t0 = time.perf_counter()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    runtime_ms = (time.perf_counter() - t0) * 1000

    return DetectionResult(
        name="SIFT",
        keypoints=list(keypoints),
        descriptors=descriptors,
        num_keypoints=len(keypoints),
        runtime_ms=runtime_ms,
    )


# --------------------------------------------------------------------------
# Convenience: run all three at once
# --------------------------------------------------------------------------
def detect_all(image: np.ndarray, **kwargs) -> dict:
    """Run Harris, ORB, and SIFT on the same image and return a dict of results."""
    return {
        "Harris": detect_harris(image, **kwargs.get("harris", {})),
        "ORB": detect_orb(image, **kwargs.get("orb", {})),
        "SIFT": detect_sift(image, **kwargs.get("sift", {})),
    }
