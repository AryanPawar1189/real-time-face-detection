"""
Static Image Face Detection Utility
=====================================
Batch-process images or single files with the same pipeline
used in real-time mode.
"""

import cv2
import numpy as np
import argparse
import logging
from pathlib import Path
from face_detector import FaceDetector, DetectionConfig, ImagePreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def process_image(
    image_path: str,
    output_path: str | None = None,
    low_light: bool = False,
    show: bool = True,
) -> dict:
    """
    Run the full detection pipeline on a single image.

    Returns a dict with:
        path, faces_detected, output_path
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    config = DetectionConfig()
    detector = FaceDetector(config)
    preprocessor = ImagePreprocessor(config)

    display, gray = preprocessor.process(img, low_light=low_light)
    detections = detector.detect(gray)
    detector._match_tracks(detections)   # seed tracks for overlay

    # Manually build simple overlay (no FPS needed for stills)
    for x, y, w, h in detections:
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display, "Face", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Landmarks
        roi_gray = gray[y:y + h, x:x + w]
        eyes = detector.eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(display,
                          (x + ex, y + ey), (x + ex + ew, y + ey + eh),
                          (255, 100, 0), 1)

    label = f"{len(detections)} face(s) detected"
    cv2.putText(display, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    saved_path = None
    if output_path:
        saved_path = output_path
    else:
        p = Path(image_path)
        saved_path = str(p.parent / f"{p.stem}_detected{p.suffix}")

    cv2.imwrite(saved_path, display)
    logger.info(f"Saved: {saved_path}")

    if show:
        cv2.imshow("Face Detection Result", display)
        logger.info("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "path": image_path,
        "faces_detected": len(detections),
        "output_path": saved_path,
    }


def batch_process(folder: str, low_light: bool = False) -> list[dict]:
    """Process all supported images in a directory."""
    results = []
    folder_path = Path(folder)
    images = [f for f in folder_path.iterdir() if f.suffix.lower() in SUPPORTED]

    if not images:
        logger.warning(f"No supported images found in {folder}")
        return results

    logger.info(f"Processing {len(images)} images from {folder}")
    out_dir = folder_path / "detected"
    out_dir.mkdir(exist_ok=True)

    for img_path in images:
        try:
            out = str(out_dir / img_path.name)
            result = process_image(str(img_path), output_path=out,
                                   low_light=low_light, show=False)
            results.append(result)
            logger.info(f"  {img_path.name}: {result['faces_detected']} face(s)")
        except Exception as e:
            logger.error(f"  Failed: {img_path.name} — {e}")

    total_faces = sum(r["faces_detected"] for r in results)
    logger.info(f"\nBatch complete: {len(results)} images, {total_faces} total faces.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static Image Face Detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path to a single image")
    group.add_argument("--folder", help="Path to a folder of images")
    parser.add_argument("--low-light", action="store_true")
    parser.add_argument("--no-show",   action="store_true", help="Don't display result (useful in headless)")
    parser.add_argument("--output",    help="Output path for single image mode")
    args = parser.parse_args()

    if args.image:
        result = process_image(args.image, args.output,
                               low_light=args.low_light, show=not args.no_show)
        print(f"\n✅  Detected {result['faces_detected']} face(s)  →  {result['output_path']}")
    else:
        batch_process(args.folder, low_light=args.low_light)
