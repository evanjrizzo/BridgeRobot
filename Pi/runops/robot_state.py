#!/usr/bin/env python3
"""
robot_state.py

Persistent storage for the robot's last known user-frame position
(rob_pos_user). All motion programs (orchestrator, autommulti2,
oneoff_cmd, etc.) should use this as the single source of truth.

We store a simple JSON file:

  {
    "rob_pos_user": [x, y, z, w, p, r]
  }

If the file is missing or corrupted, a caller-provided default is used.
"""

import json
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "robot_state.json"


def load_last_pose(default: Optional[List[float]] = None) -> List[float]:
    """
    Load the last known rob_pos_user from disk.

    If the file does not exist or is invalid, returns the provided
    default (or a 6-zero vector if no default is given).
    """
    if default is None:
        default = [400, 400, 70, 180, 0.0, 90]

    if not STATE_FILE.exists():
        return list(default)

    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return list(default)

    pose = data.get("rob_pos_user")
    if not isinstance(pose, list) or len(pose) != 6:
        return list(default)

    try:
        return [float(v) for v in pose]
    except Exception:
        return list(default)


def save_last_pose(pose: List[float]) -> None:
    """
    Save the last known rob_pos_user to disk. Expects a list of 6 floats.
    """
    if not isinstance(pose, list) or len(pose) != 6:
        raise ValueError(f"rob_pos_user must be a list of 6 floats, got {pose!r}")

    payload = {"rob_pos_user": [float(v) for v in pose]}

    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
