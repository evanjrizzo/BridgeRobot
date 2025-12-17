#!/usr/bin/env python3
"""
queue_io.py

Shared helpers for:
- loading home / named positions from home_positions.json
- loading/saving the motion queue in command_queue.json
- logging queues to command_queue_log.jsonl
"""

import json
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent

HOME_POS_FILE = BASE_DIR / "home_positions.json"
QUEUE_FILE = BASE_DIR / "command_queue.json"
QUEUE_LOG_FILE = BASE_DIR / "command_queue_log.jsonl"


def load_home_positions() -> Dict[str, Any]:
    if not HOME_POS_FILE.exists():
        raise FileNotFoundError(f"Missing {HOME_POS_FILE}")
    with HOME_POS_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("positions", {})
    return data


def get_position(name: str) -> List[float]:
    data = load_home_positions()
    positions = data.get("positions", {})
    try:
        coords = positions[name]
    except KeyError:
        raise KeyError(f"Named position {name!r} not found in home_positions.json")
    if len(coords) != 6:
        raise ValueError(f"Position {name!r} must have 6 values, got {coords!r}")
    return [float(c) for c in coords]


def get_beam_middle_x(default: float = 500.0) -> float:
    try:
        data = load_home_positions()
        return float(data.get("beam_middle_x", default))
    except Exception:
        return float(default)


def get_initial_robpos(default=None) -> List[float]:
    if default is None:
        default = [400.0, 400.0, 100.0, 180.0, 0.0, 90.0]
    try:
        data = load_home_positions()
        coords = data.get("initial_robpos_user", default)
        if len(coords) != 6:
            return list(default)
        return [float(c) for c in coords]
    except Exception:
        return list(default)


def load_queue() -> List[Dict[str, Any]]:
    if not QUEUE_FILE.exists():
        return []
    with QUEUE_FILE.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        queue = data.get("queue", [])
    else:
        queue = data
    return list(queue)


def save_queue(queue: List[Dict[str, Any]]) -> None:
    payload: Dict[str, Any] = {"queue": queue}
    with QUEUE_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_queue_log(queue: List[Dict[str, Any]]) -> None:
    if not queue:
        return
    record = {
        "queue": queue,
    }
    with QUEUE_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def clear_queue() -> None:
    save_queue([])
