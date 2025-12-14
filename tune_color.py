#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def auto_calibrate_hsv(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple HSV auto-calibration using percentile statistics over whole frame.
    Assumes you point the camera at representative rust + beam surface.
    """

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.int32)

    h = pixels[:, 0]
    s = pixels[:, 1]
    v = pixels[:, 2]

    # Percentile-based sampling
    h_min = int(np.percentile(h, 5))
    h_max = int(np.percentile(h, 95))
    s_min = int(np.percentile(s, 5))
    s_max = int(np.percentile(s, 95))
    v_min = int(np.percentile(v, 5))
    v_max = int(np.percentile(v, 95))

    # Expand bounds slightly
    pad_h = 5
    pad_s = 10
    pad_v = 10

    lower = np.array(
        [
            max(0, h_min - pad_h),
            max(0, s_min - pad_s),
            max(0, v_min - pad_v),
        ],
        dtype=np.uint8,
    )

    upper = np.array(
        [
            min(179, h_max + pad_h),
            min(255, s_max + pad_s),
            min(255, v_max + pad_v),
        ],
        dtype=np.uint8,
    )

    return lower, upper


def grab_frame_from_camera(cam_index: int, width: int, height: int):
    print(f"[Tune] Opening camera index {cam_index}...")
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[Tune] ERROR: Could not open camera {cam_index}.", file=sys.stderr)
        return None

    # Real scan program uses 1280x720 (W,H)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Warmup
    for _ in range(5):
        cap.read()

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("[Tune] ERROR: Failed to read frame.", file=sys.stderr)
        return None

    print("[Tune] Captured 1280×720 frame successfully.")
    return frame


def load_frame_from_image(path: Path):
    if not path.exists():
        print(f"[Tune] ERROR: Image not found: {path}", file=sys.stderr)
        return None

    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[Tune] ERROR: Could not read: {path}", file=sys.stderr)
        return None

    print(f"[Tune] Loaded image: {path}")
    return frame


def save_debug_images(frame_bgr, lower, upper, out_dir: Path, tag: str):
    out_dir.mkdir(exist_ok=True)

    input_path = out_dir / f"{tag}_input.png"
    mask_path = out_dir / f"{tag}_mask.png"
    overlay_path = out_dir / f"{tag}_overlay.png"

    cv2.imwrite(str(input_path), frame_bgr)
    print(f"[Tune] Saved input frame → {input_path}")

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imwrite(str(mask_path), mask)
    print(f"[Tune] Saved mask → {mask_path}")

    overlay = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
    cv2.imwrite(str(overlay_path), overlay)
    print(f"[Tune] Saved overlay → {overlay_path}")


def main():
    parser = argparse.ArgumentParser(
        description="HSV tuning tool matching the real scan resolution (1280x720)."
    )
    parser.add_argument("--image", type=str, help="Optional image path.")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--width", type=int, default=1280,
                        help="Camera width (default 1280).")
    parser.add_argument("--height", type=int, default=720,
                        help="Camera height (default 720).")
    parser.add_argument("--tag", type=str, default="calib",
                        help="Output filename tag.")

    args = parser.parse_args()

    if args.image:
        frame = load_frame_from_image(Path(args.image))
        if frame is None:
            sys.exit(1)
    else:
        frame = grab_frame_from_camera(args.cam, args.width, args.height)
        if frame is None:
            sys.exit(1)

    print("[Tune] Computing HSV bounds…")
    lower, upper = auto_calibrate_hsv(frame)

    out_dir = Path("colorcalib")
    save_debug_images(frame, lower, upper, out_dir, args.tag)

    txt_path = out_dir / f"{args.tag}_thresholds.txt"
    with txt_path.open("w") as f:
        f.write("Suggested HSV thresholds:\n")
        f.write(f"lower = {lower.tolist()}\n")
        f.write(f"upper = {upper.tolist()}\n")

    print("\n[Tune] Suggested HSV thresholds:")
    print(f"  lower = {lower.tolist()}")
    print(f"  upper = {upper.tolist()}")
    print("\nCheck colorcalib/ output for mask + overlay.\n")


if __name__ == "__main__":
    main()
