#!/usr/bin/env python3
"""
solve_homography_left.py

Reads:
  - the calibrate_tags.py output JSON (pixel coords)
  - a world_points_left.json file you edit by hand (robot X,Y)

Then:
  - matches by tag ID
  - computes a homography H such that [X, Y, 1]^T ∝ H · [u, v, 1]^T
  - saves H to homography_left.json
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
        print("Usage: solve_homography_left.py calib_tags.json world_points_left.json",
              file=sys.stderr)
        sys.exit(1)

    calib_path = sys.argv[1]
    world_path = sys.argv[2]

    calib = load_json(calib_path)
    world = load_json(world_path)

    # Build maps id -> (u,v) and id -> (X,Y)
    pix_by_id = {t["id"]: (t["u"], t["v"]) for t in calib["tags"]}
    world_by_id = {p["id"]: (p["X"], p["Y"]) for p in world["points"]}

    common_ids = sorted(set(pix_by_id.keys()) & set(world_by_id.keys()))
    if len(common_ids) < 4:
        print(f"[Error] Need at least 4 common IDs; found {len(common_ids)}: {common_ids}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[Info] Using IDs: {common_ids}")

    pts_img = []
    pts_world = []
    for tid in common_ids:
        u, v = pix_by_id[tid]
        X, Y = world_by_id[tid]
        pts_img.append([float(u), float(v)])
        pts_world.append([float(X), float(Y)])

    pts_img = np.asarray(pts_img, dtype=np.float32)
    pts_world = np.asarray(pts_world, dtype=np.float32)

    # Compute homography: image -> world
    H, mask = cv2.findHomography(pts_img, pts_world, method=0)
    if H is None:
        print("[Error] Homography computation failed.", file=sys.stderr)
        sys.exit(1)

    print("[Info] Homography matrix H (image -> world):")
    print(H)

    out = {
        "side": calib.get("side", "left"),
        "image_width": calib.get("image_width"),
        "image_height": calib.get("image_height"),
        "ids_used": common_ids,
        "H": H.tolist()
    }

    out_path = Path("homography_left.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] Saved homography to {out_path}")


if __name__ == "__main__":
    main()
