#!/usr/bin/env python3
import requests
from pathlib import Path
import sys
import time

# Tailscale IP + port of your home server's API
# If you later proxy /api via nginx on port 80, change this to "http://100.123.128.63"
SERVER = "http://100.123.128.63:8080"
API = f"{SERVER}/api"

# Root where Orchestrator creates per-run directories:
#   /tmp/bridge_robot/run_YYYYMMDD_HHMMSS/
RUN_ROOT = Path("/tmp/bridge_robot")


def log(msg: str) -> None:
    print(f"[uploader] {msg}", flush=True)


def find_latest_run_dir() -> Path:
    """
    Find the most recently modified run directory under RUN_ROOT.
    Expected pattern: run_YYYYMMDD_HHMMSS
    """
    if not RUN_ROOT.exists():
        log(f"ERROR: Run root {RUN_ROOT} does not exist.")
        sys.exit(1)

    candidates = [
        d for d in RUN_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]

    if not candidates:
        log(f"ERROR: No run_* directories found under {RUN_ROOT}")
        sys.exit(1)

    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    log(f"Using latest run directory: {latest}")
    return latest


def ensure_files_exist(run_dir: Path) -> dict:
    """
    Ensure the expected media files exist in the run directory.
    Returns a dict of file paths.
    """
    files = {
        "before": run_dir / "before.jpg",
        "after": run_dir / "after.jpg",
        "video": run_dir / "run.mp4",
    }

    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        log(f"ERROR: Missing files in {run_dir}: {missing}")
        sys.exit(1)

    return files


def create_run() -> int:
    log("Creating run on server...")
    r = requests.post(f"{API}/runs", timeout=10)
    r.raise_for_status()
    data = r.json()
    run_id = data["run_id"]
    log(f"Created run ID {run_id}")
    return run_id


def upload_capture(run_id: int, phase: str, path: Path) -> None:
    """
    Upload a still image (before/after).
    We keep side='left' just to satisfy the schema.
    """
    side = "left"
    log(f"Uploading {phase} image from {path}")
    with path.open("rb") as f:
        files = {"file": (path.name, f, "image/jpeg")}
        data = {"side": side, "phase": phase}
        r = requests.post(
            f"{API}/runs/{run_id}/capture",
            data=data,
            files=files,
            timeout=30,
        )
        r.raise_for_status()
    log(f"Uploaded {phase} image")


def upload_video(run_id: int, path: Path) -> None:
    log(f"Uploading video from {path}")
    with path.open("rb") as f:
        files = {"file": (path.name, f, "video/mp4")}
        r = requests.post(
            f"{API}/runs/{run_id}/video",
            files=files,
            timeout=300,  # longer timeout for full run video
        )
        r.raise_for_status()
    log("Uploaded video")


def cleanup(run_dir: Path, files: dict) -> None:
    log(f"Cleaning up files in {run_dir}...")
    for name, path in files.items():
        try:
            path.unlink()
            log(f"Deleted {path}")
        except FileNotFoundError:
            log(f"Already deleted: {path}")
        except Exception as e:
            log(f"Failed to delete {path}: {e}")


def main() -> None:
    run_dir = find_latest_run_dir()
    files = ensure_files_exist(run_dir)

    try:
        run_id = create_run()
        upload_capture(run_id, "before", files["before"])
        upload_capture(run_id, "after", files["after"])
        upload_video(run_id, files["video"])
    except Exception as e:
        log(f"FATAL: Upload failed: {e}")
        sys.exit(1)

    cleanup(run_dir, files)


if __name__ == "__main__":
    main()
