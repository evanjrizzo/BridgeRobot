#!/usr/bin/env python3
"""
oneoff_cmd.py

Single-command motion helper.

- Reads last known rob_pos_user from robot_state.json
- Lets the user choose a named position from home_positions.json
  or enter a custom XYZWPR.
- Asks for motion type (movel/movej, default=movel).
- Builds a safe motion sequence including cross-plane bridging
  and I-beam protection if the neutral plane must be crossed.
- Applies near-surface safety (Z<60), Y band [100,500], and
  orientation rule:
      W=180, P=0, R in {0,180} for Z<60
- Unsafe moves require explicit 'override' and abort if not given.
- Sends the sequence over TCP to the FANUC controller.
- Updates and persists rob_pos_user after every step.
- Waits up to 30 s for a final "Ready for command" at the end.
"""

import socket
import time
from typing import List, Dict, Any

from automulti2 import (
    ROBOT_IP,
    ROBOT_PORT,
    SOCKET_TIMEOUT,
    BEAM_MIDDLE_X,
    RIGHT_HOME_PARTIAL,
    RIGHT_HOME_REAL,
    LEFT_HOME_PARTIAL,
    LEFT_HOME_REAL,
    send_single_command,
)
from queue_io import load_home_positions
from robot_state import load_last_pose, save_last_pose


def _prompt_motion() -> str:
    raw = input("Motion type [movel/movej, default=movel]: ").strip().lower()
    if raw not in ("movel", "movej"):
        return "movel"
    return raw


def _prompt_named_position(names: List[str]) -> List[float]:
    print("\nNamed positions in home_positions.json:")
    for idx, name in enumerate(names, start=1):
        print(f"  {idx}. {name}")
    while True:
        sel = input("Choose index, or type name: ").strip()
        if not sel:
            continue
        if sel.isdigit():
            i = int(sel)
            if 1 <= i <= len(names):
                name = names[i - 1]
                break
            print("Index out of range.")
            continue
        if sel in names:
            name = sel
            break
        print("Unknown name; try again.")

    data = load_home_positions()
    coords = data["positions"][name]
    if len(coords) != 6:
        raise ValueError(f"Named position {name!r} must have 6 elements.")
    return [float(c) for c in coords]


def _prompt_custom_coord() -> List[float]:
    print("\nEnter custom XYZWPR as 6 space-separated numbers.")
    while True:
        text = input("X Y Z W P R: ").strip()
        parts = text.split()
        if len(parts) != 6:
            print("Need exactly 6 numbers.")
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            print("All values must be numeric.")
            continue
        return vals


def _near_surface_safety_check(
    target_pose: List[float],
    context: str,
) -> bool:
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


def _prompt_override() -> bool:
    ans = input(
        "Type 'override' to execute this oneoff move anyway, or anything else to abort: "
    ).strip().lower()
    return ans == "override"


def _build_sequence(
    current_pose: List[float],
    target_pose: List[float],
    verb: str,
) -> List[Dict[str, Any]]:
    """
    Build a safe motion sequence from current_pose to target_pose,
    including neutral-plane crossing and I-beam precautions.
    """
    seq: List[Dict[str, Any]] = []

    cx, cy, cz, cw, cp, cr = current_pose
    tx, ty, tz, tw, tp, tr = target_pose

    current_x = cx
    next_x = tx

    crossing_r_to_l = (current_x > BEAM_MIDDLE_X) and (next_x < BEAM_MIDDLE_X)
    crossing_l_to_r = (current_x < BEAM_MIDDLE_X) and (next_x > BEAM_MIDDLE_X)
    target_inside_beam = (tz < 60.0)

    if not (crossing_r_to_l or crossing_l_to_r):
        seq.append({"verb": verb, "coords": list(target_pose)})
        return seq

    if crossing_r_to_l:
        # Right -> Left
        if cz < 60.0:
            r_exit = list(RIGHT_HOME_PARTIAL)
            r_exit[5] = cr
            seq.append({"verb": "movel", "coords": r_exit})

        seq.append({"verb": "movel", "coords": list(RIGHT_HOME_PARTIAL)})
        seq.append({"verb": "movel", "coords": list(RIGHT_HOME_REAL)})
        seq.append({"verb": "movej", "coords": list(LEFT_HOME_REAL)})

        if target_inside_beam:
            l_partial = list(LEFT_HOME_PARTIAL)
            l_partial[5] = tr
            seq.append({"verb": "movel", "coords": l_partial})

        seq.append({"verb": verb, "coords": list(target_pose)})
        return seq

    # crossing_l_to_r
    if cz < 60.0:
        l_exit = list(LEFT_HOME_PARTIAL)
        l_exit[5] = cr
        seq.append({"verb": "movel", "coords": l_exit})

    seq.append({"verb": "movel", "coords": list(LEFT_HOME_PARTIAL)})
    seq.append({"verb": "movel", "coords": list(LEFT_HOME_REAL)})
    seq.append({"verb": "movej", "coords": list(RIGHT_HOME_REAL)})

    if target_inside_beam:
        r_partial = list(RIGHT_HOME_PARTIAL)
        r_partial[5] = tr
        seq.append({"verb": "movel", "coords": r_partial})

    seq.append({"verb": verb, "coords": list(target_pose)})
    return seq


def _wait_for_ready_after_sequence(sock: socket.socket, timeout: float = 30.0) -> bool:
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


def main() -> None:
    data = load_home_positions()
    names = sorted(data.get("positions", {}).keys())
    if not names:
        print("No named positions found in home_positions.json")
        return

    default_pose = data.get("initial_robpos_user", [400.0, 400.0, 100.0, 180.0, 0.0, 90.0])
    try:
        default_pose = [float(v) for v in default_pose]
    except Exception:
        default_pose = [400.0, 400.0, 100.0, 180.0, 0.0, 90.0]

    current_pose = load_last_pose(default=default_pose)

    print("One-off robot motion.")
    print(f"Last known rob_pos_user: {', '.join(f'{v:.2f}' for v in current_pose)}")

    mode = input("Use [n]amed position or [c]ustom XYZWPR? [n/c, default=n]: ").strip().lower()
    if mode not in ("c", "custom"):
        target_pose = _prompt_named_position(names)
    else:
        target_pose = _prompt_custom_coord()

    verb = _prompt_motion()
    print(f"\nAbout to send {verb.upper()} to {', '.join(f'{v:.3f}' for v in target_pose)}")

    sequence = _build_sequence(current_pose, target_pose, verb)

    # Safety checks (non-spray) with single override gate
    for idx, step in enumerate(sequence, start=1):
        coords = step["coords"]
        if not _near_surface_safety_check(coords, context=f"Step {idx} ({step['verb'].upper()})"):
            if not _prompt_override():
                print("[Info] Oneoff aborted by user for safety.")
                return
            else:
                print("[Warning] Proceeding with oneoff under user override.")
                break
        current_pose = coords

    confirm = input("Final confirmation: execute this oneoff motion? [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Aborted by user.")
        return

    print(f"\nConnecting to robot at {ROBOT_IP}:{ROBOT_PORT} ...")
    sock = socket.create_connection((ROBOT_IP, ROBOT_PORT), timeout=SOCKET_TIMEOUT)
    sock.settimeout(SOCKET_TIMEOUT)
    print("[Info] Connected.")

    final_pose = sequence[-1]["coords"]

    try:
        for idx, step in enumerate(sequence, start=1):
            v = step["verb"]
            coords = step["coords"]
            print(f"[Step {idx}/{len(sequence)}] {v.upper()} {', '.join(f'{c:.3f}' for c in coords)}")
            ok = send_single_command(sock, v, coords)
            if not ok:
                print("[Info] Command failed or connection closed; stopping sequence.")
                return
            save_last_pose(coords)

        print("[Info] Sequence finished; waiting for robot 'Ready for command' (up to 30s)...")
        ready_again = _wait_for_ready_after_sequence(sock, timeout=30.0)
        if not ready_again:
            print(
                "[Warning] Did not see 'Ready for command' within 30s after sequence.\n"
                "          Saved final target pose as rob_pos_user, but verify robot status."
            )
        else:
            print("[Info] Robot signaled 'Ready for command' again.")

        save_last_pose(final_pose)
        print("[Info] Oneoff completed; last pose persisted.")

    finally:
        try:
            sock.close()
        except Exception:
            pass
        print("[Info] Disconnected from robot.")


if __name__ == "__main__":
    main()
