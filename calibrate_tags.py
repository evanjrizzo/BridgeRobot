#!/usr/bin/env python3
"""
calibrate_tags.py

Capture a frame from LEFT or RIGHT camera, detect AprilTags,
and save an annotated image plus a JSON file with the top-left
corner pixel coordinates for each detected tag.

Usage examples:
  python3 calibrate_tags.py --side left
  python3 calibrate_tags.py --side right
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import Config   # ONLY used to read cam_index / cam2_index


def _make_aruco_detector():
    """
    Create an AprilTag detector using OpenCV's aruco module.
    Uses the tag36h11 family.
    """
    if not hasattr(cv2, "aruco"):
        print("[Error] OpenCV was built without aruco module (need opencv-contrib-python).",
              file=sys.stderr)
        sys.exit(1)

    aruco = cv2.aruco

    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    except AttributeError:
        dictionary = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)

    # New API
    if hasattr(aruco, "ArucoDetector"):
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, params)

        def detect(gray):
            return detector.detectMarkers(gray)

        return detect

    # Legacy API
    params = aruco.DetectorParameters_create()

    def detect(gray):
        return aruco.detectMarkers(gray, dictionary, parameters=params)

    return detect


def _find_top_left_corner(corners: np.ndarray) -> tuple[int, int]:
    """
    Given a (4, 2) array of corner coordinates, return the
    image-space top-left corner.

    We define "top-left" as the corner with the smallest (x + y),
    which works even if the tag is rotated.
    """
    scores = corners[:, 0] + corners[:, 1]
    idx = int(np.argmin(scores))
    x, y = corners[idx]
    return int(round(x)), int(round(y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--side",
        choices=["left", "right"],
        required=True,
        help="Which camera to use (left = cfg.cam_index, right = cfg.cam2_index)."
    )
    parser.add_argument(
        "--output-prefix",
        default="calib",
        help="Prefix for output files."
    )
    args = parser.parse_args()

    cfg = Config.load()

    if args.side == "left":
        cam_index = cfg.cam_index
    else:
        cam_index = cfg.cam2_index

    print(f"[Info] Using {args.side.upper()} camera (index={cam_index})")

    # ---- Open camera directly ----
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[Error] Cannot open /dev/video{cam_index}", file=sys.stderr)
        sys.exit(1)
    # *** Force the same settings as the main camera pipeline ***
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_h)
    if getattr(cfg, "req_fps", None):
        cap.set(cv2.CAP_PROP_FPS, cfg.req_fps)
    if getattr(cfg, "use_mjpg", False):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        
    # Warm-up frames for exposure to settle
    frame = None
    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None:
            last = frame

    if frame is None:
        print("[Error] Failed to capture frame.", file=sys.stderr)
        cap.release()
        sys.exit(1)

    cap.release()

    # ---- Detect AprilTags ----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect = _make_aruco_detector()

    corners_list, ids, _ = detect(gray)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.output_prefix}_{args.side}_{ts}"
    img_path = Path(base + ".png")
    json_path = Path(base + ".json")

    if ids is None or len(ids) == 0:
        print("[Warn] No AprilTags detected.")
        cv2.imwrite(str(img_path), frame)
        print(f"[Info] Saved raw frame to {img_path}")
        sys.exit(0)

    ids = ids.flatten()
    annotated = frame.copy()
    results = []

    print(f"[Info] Detected tags: {list(ids)}")

    for det_idx, tag_id in enumerate(ids):
        c = corners_list[det_idx].reshape(4, 2)
        u, v = _find_top_left_corner(c)

        results.append({"id": int(tag_id), "u": u, "v": v})

        # draw all corners in red
        for (x, y) in c:
            cv2.circle(annotated, (int(x), int(y)), 4, (0, 0, 255), -1)

        # highlight selected corner in green
        cv2.circle(annotated, (u, v), 6, (0, 255, 0), -1)

        text = f"id{tag_id} ({u},{v})"
        cv2.putText(
            annotated,
            text,
            (u + 5, v - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # ---- Save outputs ----
    cv2.imwrite(str(img_path), annotated)

    payload = {
        "side": args.side,
        "timestamp": ts,
        "cam_index": cam_index,
        "image_width": int(frame.shape[1]),
        "image_height": int(frame.shape[0]),
        "tags": results,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[OK] Saved annotated image → {img_path}")
    print(f"[OK] Saved tag pixel coords → {json_path}")
    print("\nDetected tag corners:")
    for r in results:
        print(f"  id {r['id']}: (u={r['u']}, v={r['v']})")


if __name__ == "__main__":
    main()
