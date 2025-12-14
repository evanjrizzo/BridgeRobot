#!/usr/bin/env python3
"""
autommulti2.py

Low-level TCP client and generic queue executor.

- Provides send_single_command(sock, verb, coords) for other modules.
- Loads named homes / cameras / beam middle from home_positions.json via queue_io.
- Uses robot_state.json for persistent rob_pos_user.
- Executes a JSON queue (from command_queue.json) with:
    * cross-plane bridging via LEFT/RIGHT home positions
    * safe exit from I-beam using partial homes when needed
- Waits up to 30 s for a final "Ready for command" after sequence.
"""

import socket
import time
from typing import List, Dict, Any

from queue_io import (
    load_queue,
    clear_queue,
    append_queue_log,
    get_position,
    get_beam_middle_x,
    get_initial_robpos,
)
from robot_state import load_last_pose, save_last_pose


ROBOT_IP = "192.168.1.10"
ROBOT_PORT = 2000
SOCKET_TIMEOUT = 1000  # seconds

# Beam middle (neutral plane) X in user frame
BEAM_MIDDLE_X = get_beam_middle_x()

# Named positions from home_positions.json
RIGHT_HOME_PARTIAL = get_position("RIGHT_HOME_PARTIAL")
RIGHT_HOME_REAL    = get_position("RIGHT_HOME_REAL")
LEFT_HOME_PARTIAL  = get_position("LEFT_HOME_PARTIAL")
LEFT_HOME_REAL     = get_position("LEFT_HOME_REAL")
RIGHT_CAM          = get_position("RIGHT_CAM")
LEFT_CAM           = get_position("LEFT_CAM")

INITIAL_ROBPOS_USER = get_initial_robpos()


# -----------------------------
# Basic line-based I/O helpers
# -----------------------------

def _recv_line(sock: socket.socket) -> str:
    """
    Receive a single 'line' terminated by CR or LF.
    Returns '' on clean EOF.
    """
    chunks: List[bytes] = []
    while True:
        try:
            b = sock.recv(1)
        except socket.timeout:
            raise TimeoutError("Timed out waiting for data from robot")

        if not b:
            return ""

        if b in (b"\r", b"\n"):
            if chunks:
                break
            else:
                continue

        chunks.append(b)

    return b"".join(chunks).decode("ascii", errors="replace")


def _wait_for_ready(sock: socket.socket) -> bool:
    """
    Block until we see 'Ready for command' from the robot.
    Returns False if the connection is closed.
    """
    while True:
        line = _recv_line(sock)
        if line == "":
            print("[Robot] Connection closed while waiting for READY.")
            return False
        print(f"[Robot] {line}")
        if "Ready for command" in line:
            return True


def _read_result(sock: socket.socket) -> bool:
    """
    Read responses after sending a command, until we see 'OK' or 'ERR...'.
    Returns False if the connection is closed.
    """
    while True:
        try:
            line = _recv_line(sock)
        except TimeoutError:
            print("[Error] Timed out waiting for response from robot.")
            return False

        if line == "":
            print("[Robot] Connection closed.")
            return False

        print(f"[Robot] {line}")

        if line.startswith("OK") or line.startswith("ERR"):
            return True


def _wait_for_ready_after_sequence(sock: socket.socket, timeout: float = 30.0) -> bool:
    """
    After finishing a sequence, optionally wait for the robot
    to print "Ready for command" again, up to 'timeout' seconds.

    Returns True if "Ready for command" is seen, False otherwise.
    """
    sock.settimeout(1.0)
    end = time.monotonic() + timeout
    buf = b""

    while time.monotonic() < end:
        try:
            chunk = sock.recv(1)
        except socket.timeout:
            continue
        except OSError:
            return False

        if not chunk:
            return False

        if chunk in (b"\r", b"\n"):
            if not buf:
                continue
            line = buf.decode("ascii", errors="replace")
            buf = b""
            print(f"[Robot] {line}")
            if "Ready for command" in line:
                return True
            continue

        buf += chunk

    return False


# -----------------------------
# Command formatting
# -----------------------------

def _format_line(verb: str, coords) -> str:
    """
    Build a line like 'MOVEL x y z w p r\n' from a verb and coord list.
    """
    verb = verb.strip().lower()
    if verb not in ("movel", "movej"):
        raise ValueError(f"Invalid verb: {verb!r}")

    if len(coords) != 6:
        raise ValueError(f"{verb} requires 6 coords, got {len(coords)}: {coords!r}")

    try:
        nums = [float(c) for c in coords]
    except (TypeError, ValueError):
        raise ValueError(f"Non-numeric coord in {coords!r}")

    return verb.upper() + " " + " ".join(f"{n:.6f}" for n in nums) + "\n"


def send_single_command(sock: socket.socket, verb: str, coords) -> bool:
    """
    Full send cycle for a single low-level command:
    - Wait for READY
    - Send command
    - Read result (OK/ERR)
    Returns False if anything fails or connection closes.
    """
    print(f"[Info] Waiting for robot READY for {verb.upper()} ...")
    if not _wait_for_ready(sock):
        print("[Info] Robot not ready / connection closed.")
        return False

    try:
        line = _format_line(verb, coords)
    except ValueError as e:
        print(f"[Error] {e}")
        return False

    printable = line.strip()
    print(f"[Client] {printable}")
    sock.sendall(line.encode("ascii"))

    if not _read_result(sock):
        print("[Info] Communication issue during", verb.upper())
        return False

    return True


# -----------------------------
# Safety helpers (near-surface)
# -----------------------------

def _near_surface_safety_check(
    target_pose: List[float],
    spray: bool,
    context: str,
) -> bool:
    """
    Apply near-surface safety rules for a single motion endpoint.

    Returns True if the move is allowed without override for spray,
    or allowed to proceed (possibly with override) for non-spray.
    For autommulti2 (generic), we only do hard filtering for spray.
    """
    x, y, z, w, p, r = target_pose
    in_near_zone = (z < 60.0)
    in_y_band = (100.0 <= y <= 500.0)
    orientation_ok = (abs(w - 180.0) < 1e-6 and abs(p - 0.0) < 1e-6 and abs(r) in (0.0, 180.0))

    if not in_near_zone:
        return True

    if not in_y_band:
        print(
            f"[Safety] {context}: move would enter Z<60 with Y={y:.3f} "
            f"outside test band [100,500]."
        )
        return False

    if not orientation_ok:
        print(
            f"[Safety] {context}: move would enter Z<60 with disallowed orientation "
            f"W={w:.3f}, P={p:.3f}, R={r:.3f} (allowed: W=180, P=0, R=0 or 180)."
        )
        return False

    return True


# -----------------------------
# Cross-plane sequence builder
# -----------------------------

def _build_sequence_for_entry(
    current_pose: List[float],
    entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build a safe motion sequence for a queue entry, including:
    - cross-plane bridging
    - safe exit from I-beam via partial homes if needed

    Entry format:
      {
        "verb": "movel"/"movej",
        "coords": [x,y,z,w,p,r],
        "meta": {...optional...}
      }
    """
    verb = entry.get("verb", "").strip().lower()
    coords = entry.get("coords", [])
    meta = entry.get("meta", {}) or {}

    if verb not in ("movel", "movej"):
        raise ValueError(f"Invalid verb in queue: {verb!r}")
    if len(coords) != 6:
        raise ValueError(f"Queue entry coords must have length 6, got {coords!r}")

    cx, cy, cz, cw, cp, cr = current_pose
    tx, ty, tz, tw, tp, tr = map(float, coords)

    current_x = cx
    next_x = tx

    crossing_r_to_l = (current_x > BEAM_MIDDLE_X) and (next_x < BEAM_MIDDLE_X)
    crossing_l_to_r = (current_x < BEAM_MIDDLE_X) and (next_x > BEAM_MIDDLE_X)
    target_inside_beam = (tz < 60.0)

    seq: List[Dict[str, Any]] = []

    if not (crossing_r_to_l or crossing_l_to_r):
        # No cross-plane
        seq.append({"verb": verb, "coords": list(coords), "meta": meta})
        return seq

    if crossing_r_to_l:
        # Right -> Left
        if cz < 60.0:
            # Exit via right partial with same R as current
            r_exit = list(RIGHT_HOME_PARTIAL)
            r_exit[5] = cr
            seq.append({"verb": "movel", "coords": r_exit, "meta": {"source": "exit_beam_right"}})

        seq.append({"verb": "movel", "coords": list(RIGHT_HOME_PARTIAL), "meta": {"source": "cross_r_to_l"}})
        seq.append({"verb": "movel", "coords": list(RIGHT_HOME_REAL), "meta": {"source": "cross_r_to_l"}})
        seq.append({"verb": "movej", "coords": list(LEFT_HOME_REAL), "meta": {"source": "cross_r_to_l"}})

        if target_inside_beam:
            l_partial = list(LEFT_HOME_PARTIAL)
            l_partial[5] = tr
            seq.append({"verb": "movel", "coords": l_partial, "meta": {"source": "enter_beam_left"}})

        seq.append({"verb": verb, "coords": list(coords), "meta": meta})
        return seq

    # crossing_l_to_r
    if cz < 60.0:
        l_exit = list(LEFT_HOME_PARTIAL)
        l_exit[5] = cr
        seq.append({"verb": "movel", "coords": l_exit, "meta": {"source": "exit_beam_left"}})

    seq.append({"verb": "movel", "coords": list(LEFT_HOME_PARTIAL), "meta": {"source": "cross_l_to_r"}})
    seq.append({"verb": "movel", "coords": list(LEFT_HOME_REAL), "meta": {"source": "cross_l_to_r"}})
    seq.append({"verb": "movej", "coords": list(RIGHT_HOME_REAL), "meta": {"source": "cross_l_to_r"}})

    if target_inside_beam:
        r_partial = list(RIGHT_HOME_PARTIAL)
        r_partial[5] = tr
        seq.append({"verb": "movel", "coords": r_partial, "meta": {"source": "enter_beam_right"}})

    seq.append({"verb": verb, "coords": list(coords), "meta": meta})
    return seq


# -----------------------------
# Queue execution
# -----------------------------

def execute_queue(queue: List[Dict[str, Any]]) -> None:
    """
    Generic queue executor.

    - Loads last known rob_pos_user from robot_state.json
    - Builds safe sequences for each queue entry (cross-plane aware)
    - Applies near-surface safety checks for any spray moves:
         meta["spray"] == True -> hard filtered
      (non-spray safety/override is handled at higher levels if desired)
    - After each successful command, updates and persists rob_pos_user
    - After finishing, waits up to 30 s for "Ready for command" again
    """
    if not queue:
        print("[Info] Command queue is empty; nothing to do.")
        return

    rob_pos_user = load_last_pose(default=INITIAL_ROBPOS_USER)

    print(f"Connecting to robot at {ROBOT_IP}:{ROBOT_PORT} ...")
    sock = socket.create_connection((ROBOT_IP, ROBOT_PORT), timeout=SOCKET_TIMEOUT)
    sock.settimeout(SOCKET_TIMEOUT)
    print("[Info] Connected.")

    try:
        for idx, entry in enumerate(queue):
            print("\n======================================")
            print(f"[Step] Queue index #{idx + 1}")

            seq = _build_sequence_for_entry(rob_pos_user, entry)

            # Safety filter for spray moves only (if present)
            is_spray = bool(entry.get("meta", {}).get("spray", False))
            if is_spray:
                tx, ty, tz, tw, tp, tr = map(float, entry["coords"])
                if not _near_surface_safety_check(
                    [tx, ty, tz, tw, tp, tr],
                    spray=True,
                    context=f"Entry #{idx + 1}",
                ):
                    print("[Safety] Skipping spray move due to near-surface rule.")
                    continue

            print("[Sequence to execute:]")
            for j, step in enumerate(seq, start=1):
                v = step["verb"]
                c = step["coords"]
                preview = f"{v.upper()} " + " ".join(f"{float(n):.3f}" for n in c)
                print(f"  {j}. {preview}")

            for step in seq:
                v = step["verb"]
                c = step["coords"]
                ok = send_single_command(sock, v, c)
                if not ok:
                    print("[Info] Aborting remaining sequence due to communication error.")
                    return
                rob_pos_user = list(c)
                save_last_pose(rob_pos_user)

        print("\n[Info] All queued steps have been executed.")
        print("[Info] Waiting for final 'Ready for command' (up to 30s)...")
        ready_again = _wait_for_ready_after_sequence(sock, timeout=30.0)
        if not ready_again:
            print(
                "[Warning] Did not see 'Ready for command' within 30s after sequence.\n"
                "          Saved final pose as rob_pos_user, but verify robot status."
            )
        else:
            print("[Info] Robot signaled 'Ready for command' again.")

        save_last_pose(rob_pos_user)

    except KeyboardInterrupt:
        print("\n[Info] Keyboard interrupt received, closing connection...")

    finally:
        try:
            sock.close()
        except Exception:
            pass
        print("[Info] Disconnected from robot.")


def main() -> None:
    queue = load_queue()
    execute_queue(queue)
    append_queue_log(queue)
    clear_queue()
    print("[Info] Queue logged and cleared.")


if __name__ == "__main__":
    main()
