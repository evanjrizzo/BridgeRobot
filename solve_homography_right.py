#!/usr/bin/env python3
"""
solve_homography_right.py

Compute a planar homography for the RIGHT side camera.

Inputs:
  1) calib JSON produced by calibrate_tags.py, e.g.:
       calib_dev0_20251201_153000.json

     Format (example):
       {
         "device": 0,
         "timestamp": "...",
         "image_width": 1280,
         "image_height": 720,
         "tags": [
           { "id": 1, "u": 101, "v": 317 },
           { "id": 2, "u": 558, "v": 313 },
           ...
         ]
       }

  2) world_points_right.json: hand-edited file mapping tag IDs to
     robot user-frame X,Y coordinates for the SAME physical corners
     (laser dot on the top-left corner used in calibration).

     Example:
       {
         "points": [
           { "id": 1, "X": 480.3, "Y": 320.1 },
           { "id": 2, "X": 610.7, "Y": 318.9 },
           { "id": 4, "X": 483.5, "Y":  59.4 },
           { "id": 5, "X": 614.2, "Y":  60.2 }
         ]
       }

The script:
  - Matches IDs present in BOTH files
  - Requires at least 4 common IDs
  - Computes a homography H such that:

      [X, Y, 1]^T ∝ H · [u, v, 1]^T

  - Writes homography_right.json with H and metadata.
"""

import json
from pathlib import Path
import sys

import cv2
import numpy as np


def load_json(path: str):
    p = Path(path)
    if not p.exists():
        print(f"[Error] Missing file: {p}", file=sys.stderr)
        sys.exit(1)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "  python3 solve_homography_right.py calib_right.json world_points_right.json",
            file=sys.stderr,
        )
        sys.exit(1)

    calib_path = sys.argv[1]
    world_path = sys.argv[2]

    calib = load_json(calib_path)
    world = load_json(world_path)

    # id -> (u, v)
    try:
        pix_by_id = {int(t["id"]): (float(t["u"]), float(t["v"])) for t in calib["tags"]}
    except KeyError as e:
        print(f"[Error] calib JSON missing field: {e}", file=sys.stderr)
        sys.exit(1)

    # id -> (X, Y)
    try:
        world_by_id = {int(p["id"]): (float(p["X"]), float(p["Y"])) for p in world["points"]}
    except KeyError as e:
        print(f"[Error] world_points JSON missing field: {e}", file=sys.stderr)
        sys.exit(1)

    common_ids = sorted(set(pix_by_id.keys()) & set(world_by_id.keys()))
    if len(common_ids) < 4:
        print(
            f"[Error] Need at least 4 common tag IDs; found {len(common_ids)}: {common_ids}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[Info] Common IDs used for homography: {common_ids}")

    pts_img = []
    pts_world = []

    for tid in common_ids:
        u, v = pix_by_id[tid]
        X, Y = world_by_id[tid]
        pts_img.append([u, v])
        pts_world.append([X, Y])

    pts_img = np.asarray(pts_img, dtype=np.float32)
    pts_world = np.asarray(pts_world, dtype=np.float32)

    # Compute H: image -> world
    H, mask = cv2.findHomography(pts_img, pts_world, method=0)
    if H is None:
        print("[Error] Homography computation failed (H is None).", file=sys.stderr)
        sys.exit(1)

    print("[Info] Homography matrix H (image -> world):")
    print(H)

    # Evaluate reprojection error for sanity
    pts_world_pred = cv2.perspectiveTransform(
        pts_img.reshape(-1, 1, 2).astype(np.float32),
        H.astype(np.float64),
    ).reshape(-1, 2)

    diffs = pts_world_pred - pts_world
    errs = np.linalg.norm(diffs, axis=1)
    mean_err = float(np.mean(errs))
    max_err = float(np.max(errs))

    print(f"[Info] Reprojection error: mean={mean_err:.3f} mm, max={max_err:.3f} mm")

    out = {
        "side": "right",
        "image_width": calib.get("image_width"),
        "image_height": calib.get("image_height"),
        "device": calib.get("device"),
        "ids_used": common_ids,
        "H": H.tolist(),
        "reprojection_error_mm": {
            "mean": mean_err,
            "max": max_err,
        },
    }

    out_path = Path("homography_right.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] Saved homography to {out_path}")


if __name__ == "__main__":
    main()
