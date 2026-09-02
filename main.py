"""
main.py
-------
Command-line entry point for the Keypoint Detection and Visualization System.

Usage
-----
    python3 main.py                          # runs on the built-in synthetic sample image
    python3 main.py --image path/to/img.jpg  # runs on a user-supplied image
    python3 main.py --image img.jpg --out-dir results

Outputs (written to --out-dir, default "output/"):
    harris_keypoints.png    - Harris corners drawn on the image
    harris_heatmap.png      - Harris response map overlay
    orb_keypoints.png       - ORB keypoints (rich: size + orientation)
    sift_keypoints.png      - SIFT keypoints (rich: size + orientation)
    comparison_panel.png    - Harris | ORB | SIFT side by side
    summary_chart.png       - bar charts of keypoint counts & runtime
    summary.txt             - plain-text numeric summary
"""

import argparse
import os
import sys

import cv2

from detectors import detect_all
from visualize import (
    annotate,
    draw_harris_heatmap,
    side_by_side,
    summary_chart,
)


def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{path}'")
    return img


def run(image_path: str, out_dir: str, max_dim: int = 1000):
    os.makedirs(out_dir, exist_ok=True)

    image = load_image(image_path)

    # Keep things fast/legible: downscale very large images
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    print(f"Loaded image: {image_path}  ({image.shape[1]}x{image.shape[0]})")
    print("Running detectors: Harris, ORB, SIFT ...")

    results = detect_all(image)

    for name, result in results.items():
        print(f"  {name:8s} -> {result.num_keypoints:5d} keypoints  "
              f"({result.runtime_ms:.2f} ms)")

    # Individual annotated images
    cv2.imwrite(os.path.join(out_dir, "harris_keypoints.png"), annotate(image, results["Harris"]))
    cv2.imwrite(os.path.join(out_dir, "harris_heatmap.png"), draw_harris_heatmap(image, results["Harris"]))
    cv2.imwrite(os.path.join(out_dir, "orb_keypoints.png"), annotate(image, results["ORB"]))
    cv2.imwrite(os.path.join(out_dir, "sift_keypoints.png"), annotate(image, results["SIFT"]))

    # Comparison panel
    panel = side_by_side(image, results)
    cv2.imwrite(os.path.join(out_dir, "comparison_panel.png"), panel)

    # Chart
    summary_chart(results, os.path.join(out_dir, "summary_chart.png"))

    # Text summary
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Keypoint Detection Summary for: {image_path}\n")
        f.write(f"Image size: {image.shape[1]}x{image.shape[0]}\n")
        f.write("-" * 50 + "\n")
        for name, result in results.items():
            f.write(f"{name:8s} | keypoints: {result.num_keypoints:6d} | "
                    f"runtime: {result.runtime_ms:8.2f} ms | "
                    f"descriptor: {'None' if result.descriptors is None else result.descriptors.shape}\n")

    print(f"\nAll outputs written to: {os.path.abspath(out_dir)}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Keypoint Detection and Visualization System")
    parser.add_argument("--image", type=str, default=None,
                         help="Path to input image. If omitted, a synthetic demo image is generated.")
    parser.add_argument("--out-dir", type=str, default="output",
                         help="Directory to write results to (default: output/)")
    parser.add_argument("--max-dim", type=int, default=1000,
                         help="Max width/height in pixels before downscaling (default: 1000)")
    args = parser.parse_args()

    image_path = args.image
    if image_path is None:
        from make_sample_image import generate_scene
        os.makedirs("sample_images", exist_ok=True)
        image_path = "sample_images/synthetic_scene.png"
        cv2.imwrite(image_path, generate_scene())
        print(f"No --image supplied; generated demo image at {image_path}")

    run(image_path, args.out_dir, args.max_dim)


if __name__ == "__main__":
    sys.exit(main())
