"""
make_sample_image.py
---------------------
Generates a synthetic test image containing a mix of features that are
ideal for demonstrating corner/keypoint detectors: a checkerboard region
(strong Harris corners), assorted polygons at different scales/rotations
(good for ORB/SIFT scale & rotation invariance), and some flat / low
texture regions (should produce few or no keypoints).

Run directly to write sample_images/synthetic_scene.png, or import
`generate_scene()` to get the array in memory.
"""

import cv2
import numpy as np


def _checkerboard(size: int, square: int) -> np.ndarray:
    board = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, square):
        for j in range(0, size, square):
            if ((i // square) + (j // square)) % 2 == 0:
                board[i:i + square, j:j + square] = 255
    return board


def generate_scene(width: int = 900, height: int = 650) -> np.ndarray:
    img = np.full((height, width, 3), 235, dtype=np.uint8)  # light gray background

    # Checkerboard patch (top-left) -> classic Harris corner test pattern
    board = _checkerboard(240, 30)
    board_bgr = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    img[30:270, 30:270] = board_bgr

    # Star polygon (good multi-corner shape)
    center = (450, 150)
    pts = []
    for k in range(10):
        angle = np.pi / 5 * k - np.pi / 2
        r = 90 if k % 2 == 0 else 40
        pts.append((int(center[0] + r * np.cos(angle)), int(center[1] + r * np.sin(angle))))
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], (60, 90, 200))

    # Rotated square (tests orientation invariance)
    rect = ((680, 150), (160, 160), 30)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(img, [box], (40, 160, 40))

    # Circles at different scales (SIFT scale-space test)
    for i, r in enumerate([15, 30, 55]):
        cv2.circle(img, (150 + i * 150, 450), r, (200, 120, 30), -1)
        cv2.circle(img, (150 + i * 150, 450), r, (0, 0, 0), 2)

    # Text (lots of small high-contrast corners)
    cv2.putText(img, "KEYPOINTS", (520, 420), cv2.FONT_HERSHEY_DUPLEX, 1.6,
                (20, 20, 20), 3, cv2.LINE_AA)

    # A large flat / low-texture region (should yield ~0 keypoints)
    cv2.rectangle(img, (520, 460), (860, 610), (235, 235, 235), -1)
    cv2.rectangle(img, (520, 460), (860, 610), (210, 210, 210), 1)

    # Thin diagonal lines (edges, not corners -> good Harris/ORB contrast case)
    cv2.line(img, (30, 320), (270, 620), (0, 0, 0), 2)
    cv2.line(img, (270, 320), (30, 620), (0, 0, 0), 2)

    # Light Gaussian noise for realism
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


if __name__ == "__main__":
    scene = generate_scene()
    out_path = "sample_images/synthetic_scene.png"
    cv2.imwrite(out_path, scene)
    print(f"Wrote {out_path}  shape={scene.shape}")
