import sys
import time
import math
import json
import socket
import threading
from typing import Dict, Any, Optional, List, Tuple

import os
import subprocess

import cv2

from config import Config
from cam import Cam
from servos import Servos
from ultrason import UltraSon
from sense import Sense
from website import Website

from queue_io import get_position
from robot_state import load_last_pose, save_last_pose
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

LEFT_CAM = get_position("LEFT_CAM")
RIGHT_CAM = get_position("RIGHT_CAM")

# --- spray head correction offsets (mm), by side + row ---
# These are now all zero and NOT applied anywhere below.
RIGHT_TOP_DX = -50.0
RIGHT_TOP_DY = 100.0
RIGHT_BOT_DX = 20.0
RIGHT_BOT_DY = 30.0

LEFT_TOP_DX = -30.0  # THIS IS THE OFFSET FOR X
LEFT_TOP_DY = 20.0   # THIS IS THE OFFSET FOR Y
LEFT_BOT_DX = 20.0
LEFT_BOT_DY = 30.0


class Orchestrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Primary camera (robot view)
        self.cam = Cam(cfg, cam_index=cfg.cam_index)

        # Spray check
        self._pending_spray_meta: Optional[Dict[str, Any]] = None
        self._pending_spray_delay: float = 0.0  # dynamic wait before spray based on distance

        # --- Camera 2: dedicated capture camera (index 2) ---
        self.cam2: Optional[Cam] = None
        try:
            self.cam2 = Cam(cfg, cam_index=2)
            print("[Orchestrator] Camera 2 initialized at index 2 (media capture camera).")
        except Exception as e:
            print(f"[Camera2] FAILED to open camera index 2: {e}", file=sys.stderr)
            self.cam2 = None


        # Decide which camera is used for scanning/targeting
        self.use_cam2_for_scan = bool(getattr(cfg, "use_cam2_for_scan", False))
        if self.use_cam2_for_scan and self.cam2 is not None:
            self.scan_cam = self.cam2
            print("[Orchestrator] Using CAM2 as scan/targeting camera.")
        else:
            self.scan_cam = self.cam
            if self.use_cam2_for_scan and self.cam2 is None:
                print("[Orchestrator] use_cam2_for_scan=True, but cam2 not available; falling back to CAM0.")
            else:
                print("[Orchestrator] Using CAM0 as scan/targeting camera.")

        # JPEG buffers
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg = bytearray()

        self._jpeg2_lock = threading.Lock()
        self._latest_jpeg2 = bytearray()

        # Grab initial frame so /stream has something (from scan_cam)
        try:
            ok0, frame0 = self.scan_cam.read_frame()
            if ok0 and frame0 is not None:
                jpeg0 = self.scan_cam.encode(frame0)
                if jpeg0:
                    with self._jpeg_lock:
                        self._latest_jpeg[:] = jpeg0
        except Exception as e:
            print(f"[ScanCam] initial frame grab failed: {e}", file=sys.stderr)

        self.sense = Sense(cfg)
        self.servos = Servos(cfg)

        # Ultrasonic
        self.ultra: Optional[UltraSon] = None
        try:
            self.ultra = UltraSon(cfg)
        except Exception as e:
            print(f"[Ultrasonic] disabled: {e}", file=sys.stderr)
            self.ultra = None

        # Status for website
        self.status: Dict[str, Any] = {
            "label": "INIT",
            "fps": 0.0,
            "last_boxes": 0,
            "target_rgb": list(cfg.target_rgb),
            "lower": self.sense.lower.tolist(),
            "upper": self.sense.upper.tolist(),
            "area_min": cfg.area_min,
            "encoder": self.scan_cam.encoder_backend,
            "distance_from_bridge_in": None,
            "detection_center": None,
            "scan_active": False,
            "scan_state": "idle",  # idle | scanning_left | scanning_right | other
            "mode": "idle",        # idle | moving | spraying | error
        }

        self.website = Website(
            cfg=cfg,
            frame_provider=self.get_jpeg,
            status_provider=self.get_status,
            second_frame_provider=self.get_jpeg2,
        )

        # FPS tracking
        self._fps_prev = time.monotonic()
        self._fps_frames = 0
        self._fps_val = 0.0

        # Persistent robot pose
        default_pose = get_position("LEFT_HOME_REAL")
        self._rob_pos_user: List[float] = load_last_pose(default=default_pose)

        # --- Media capture state (NEW) ---
        self._run_dir: Optional[str] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._video_fps: float = 10.0

    # --------- NEW MEDIA HELPERS ----------

    def _ensure_run_dir(self) -> str:
        """
        Ensure we have a unique directory under /tmp/bridge_robot
        for this run, e.g. /tmp/bridge_robot/run_20251212_123456
        """
        base = "/tmp/bridge_robot"
        try:
            os.makedirs(base, exist_ok=True)
        except Exception as e:
            print(f"[Media] Could not create base dir {base}: {e}", file=sys.stderr)

        if self._run_dir:
            return self._run_dir

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._run_dir = os.path.join(base, f"run_{ts}")
        try:
            os.makedirs(self._run_dir, exist_ok=True)
        except Exception as e:
            print(f"[Media] Could not create run dir {self._run_dir}: {e}", file=sys.stderr)
        else:
            print(f"[Media] Using run dir {self._run_dir}")
        return self._run_dir

    def _capture_still(self, name: str) -> None:
        """
        Grab a single frame from scan_cam and save as <name>.jpg
        into the current run directory.
        """
        if self.scan_cam is None:
            return

        if self.cam2 is None:
            print(f"[Media] Failed to grab frame for still {name}", file=sys.stderr)
            return
            
        ok, frame = self.cam2.read_frame()

        run_dir = self._ensure_run_dir()
        path = os.path.join(run_dir, f"{name}.jpg")
        try:
            cv2.imwrite(path, frame)
            print(f"[Media] Saved still {name} -> {path}")
        except Exception as e:
            print(f"[Media] Failed to save still {name}: {e}", file=sys.stderr)

    def _start_run_recording(self) -> None:
        """
        Initialize a VideoWriter for this run and write the first frame.
        """
        if self._video_writer is not None:
            # Already recording
            return

        if self.scan_cam is None:
            return

        run_dir = self._ensure_run_dir()

        if self.cam2 is None:
            print("[Media] Could not grab frame to initialize VideoWriter", file=sys.stderr)
            return

        ok, frame = self.cam2.read_frame()

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        path = os.path.join(run_dir, "run.mp4")
        writer = cv2.VideoWriter(path, fourcc, self._video_fps, (w, h))
        if not writer.isOpened():
            print(f"[Media] Failed to open VideoWriter at {path}", file=sys.stderr)
            return

        self._video_writer = writer
        try:
            self._video_writer.write(frame)
        except Exception as e:
            print(f"[Media] Error writing first frame to video: {e}", file=sys.stderr)
        else:
            print(f"[Media] Started run recording -> {path}")

    def _record_frame(self, frame) -> None:
        """
        Append a frame to the current run video, if recording is active.
        """
        writer = self._video_writer
        if writer is None:
            return
        try:
            writer.write(frame)
        except Exception as e:
            print(f"[Media] Error writing frame to video: {e}", file=sys.stderr)

    def _stop_run_recording(self) -> None:
        """
        Stop and release the current run video writer.
        """
        writer = self._video_writer
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        self._video_writer = None
        print("[Media] Stopped run recording.")

    def _upload_run_media(self) -> None:
        """
        Fire-and-forget: launch send_to_server.py in the background.
        The script is responsible for finding the latest run under
        /tmp/bridge_robot and uploading its media.
        """
        try:
            print("[Media] Launching send_to_server.py for latest run.")
            subprocess.Popen(
                ["python3", "send_to_server.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[Media] Failed to launch uploader: {e}", file=sys.stderr)

    # --------- END MEDIA HELPERS ----------

    # Shield helpers (delegate to Servos)

    def _shield_open(self) -> None:
        try:
            self.servos.open_shield()
        except Exception as e:
            print(f"[Servos] shield_open failed: {e}", file=sys.stderr)

    def _shield_close(self) -> None:
        try:
            self.servos.close_shield()
        except Exception as e:
            print(f"[Servos] shield_close failed: {e}", file=sys.stderr)

    # HTTP providers

    def get_jpeg(self) -> bytes:
        with self._jpeg_lock:
            return bytes(self._latest_jpeg)

    def get_jpeg2(self) -> bytes:
        with self._jpeg2_lock:
            return bytes(self._latest_jpeg2)

    def get_status(self) -> Dict[str, Any]:
        if self.ultra is not None:
            self.status["distance_from_bridge_in"] = self.ultra.latest_inches()
        else:
            self.status["distance_from_bridge_in"] = None
        return dict(self.status)

    # Camera2 loop

    def _cam2_loop(self) -> None:
        if self.cam2 is None:
            return
        self._cam2_running = True
        try:
            while self._cam2_running:
                ok2, frame2 = self.cam2.read_frame()
                if not ok2 or frame2 is None:
                    time.sleep(0.01)
                    continue
                jpeg2 = self.cam2.encode(frame2)
                if jpeg2:
                    with self._jpeg2_lock:
                        self._latest_jpeg2[:] = jpeg2
        except Exception as e:
            print(f"[Camera2] loop error: {e}", file=sys.stderr)
        finally:
            self._cam2_running = False

    # Safety

    @staticmethod
    def _near_surface_safety_check(
        coords: List[float],
        spray: bool,
        context: str,
    ) -> bool:
        x, y, z, w, p, r = coords
        in_near_zone = (z < 60.0)
        # Updated Y band for near-surface validity
        in_y_band = (160.0 <= y <= 570.0)
        # Orientation enforcement: W=180, P=0, R=90 (fixed J6 orientation)
        orientation_ok = (
            abs(w - 180.0) < 1e-6 and
            abs(p - 0.0) < 1e-6 and
            abs(r - 90.0) < 1e-6
        )

        if not in_near_zone:
            return True

        if not in_y_band:
            print(
                f"[Safety] {context}: move would enter Z<60 with Y={y:.3f} "
                f"outside test band [200,530]."
            )
            return False

        if not orientation_ok:
            print(
                f"[Safety] {context}: move would enter Z<60 with disallowed orientation "
                f"W={w:.3f}, P={p:.3f}, R={r:.3f} (allowed: W=180, P=0, R=90)."
            )
            return False

        return True

    @staticmethod
    def _merge_targets(
        targets: List[Dict[str, Any]],
        max_center_dist_mm: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """
        Upstream, _dedupe_detections() already merges detections that are
        within ~10 mm in world space across frames of the scan.

        Here we ONLY collapse truly duplicate targets (within a small
        distance threshold). We do NOT merge by bounding-box overlap
        anymore, so subtargets that represent different positions along
        the same blob remain separate spray targets.
        """
        if not targets:
            return []

        unique: List[Dict[str, Any]] = []

        for t in targets:
            x, y, z, w, p, r = t["xyzwpr"]
            merged = False

            for u in unique:
                ux, uy, uz, uw, up, ur = u["xyzwpr"]
                if math.hypot(x - ux, y - uy) <= max_center_dist_mm:
                    # Average nearly-identical poses together
                    u["xyzwpr"][0] = (ux + x) / 2.0
                    u["xyzwpr"][1] = (uy + y) / 2.0
                    u["xyzwpr"][2] = (uz + z) / 2.0
                    merged = True
                    break

            if not merged:
                # Clone so later modifications don't affect the original list
                unique.append(dict(t))

        return unique

    def _build_queue_for_side(
        self,
        detections: List[Dict[str, Any]],
        side: str,
    ) -> List[Dict[str, Any]]:
        side = side.lower()
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

        if not detections:
            return []

        home_partial = list(LEFT_HOME_PARTIAL if side == "left" else RIGHT_HOME_PARTIAL)
        home_real = list(LEFT_HOME_REAL if side == "left" else RIGHT_HOME_REAL)

        merged = self._merge_targets(detections, max_center_dist_mm=5.0)
        filtered: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []

        for t in merged:
            y = t["xyzwpr"][1]
            if y < 160.0 or y > 570.0:
                print(f"[Safety] Skipping spray target Y={y:.3f} outside [275,530].")
                skipped.append({"reason": "y_out_of_band", "xyzwpr": t["xyzwpr"]})
                continue

            if not self._near_surface_safety_check(
                t["xyzwpr"],
                spray=True,
                context="queue_build",
            ):
                print("[Safety] Skipping spray target due to near-surface rule.")
                skipped.append({"reason": "near_surface_reject", "xyzwpr": t["xyzwpr"]})
                continue

            filtered.append(t)

        if skipped:
            log_name = f"skipped_targets_{side}.json"
            try:
                with open(log_name, "w", encoding="utf-8") as f:
                    json.dump({"skipped": skipped}, f, indent=2)
                print(f"[{side.capitalize()}] Logged {len(skipped)} skipped targets to {log_name}")
            except Exception as e:
                print(f"[{side.capitalize()}] Could not write {log_name}: {e}", file=sys.stderr)

        if not filtered:
            return []

        top = [t for t in filtered if t["row"] == "top"]
        bot = [t for t in filtered if t["row"] == "bottom"]

        top_sorted = sorted(top, key=lambda t: t["xyzwpr"][0])
        bot_sorted = sorted(bot, key=lambda t: t["xyzwpr"][0], reverse=True)

        queue: List[Dict[str, Any]] = []

        # First do all top-row targets (in X ascending)
        for t in top_sorted:
            queue.append({
                "verb": "movel",
                "coords": t["xyzwpr"],
                "meta": {"side": side, "row": "top", "source": "sense", "spray": True},
            })

        # Then bottom-row targets (in X descending)
        for t in bot_sorted:
            queue.append({
                "verb": "movel",
                "coords": t["xyzwpr"],
                "meta": {"side": side, "row": "bottom", "source": "sense", "spray": True},
            })

        # Exit beam & go home
        last_z = queue[-1]["coords"][2]
        last_r = queue[-1]["coords"][5]
        if last_z < 60.0:
            exit_partial = list(home_partial)
            if len(exit_partial) == 6:
                exit_partial[5] = last_r
            queue.append({
                "verb": "movel",
                "coords": exit_partial,
                "meta": {"side": side, "row": "home", "source": "exit_beam", "spray": False},
            })

        queue.append({
            "verb": "movel",
            "coords": home_real,
            "meta": {"side": side, "row": "home", "source": "return_home", "spray": False},
        })

        return queue

    def _dedupe_detections(
        self,
        detections: List[Dict[str, Any]],
        xy_tol_mm: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        Merge detections that are effectively the same world-space spot.

        Any detections whose (x, y) are within xy_tol_mm of an existing one
        are averaged into a single entry. All other fields are left as-is
        from the first detection that created the cluster.
        """
        unique: List[Dict[str, Any]] = []

        for d in detections:
            x = float(d.get("x", 0.0))
            y = float(d.get("y", 0.0))

            merged = False
            for u in unique:
                ux = float(u.get("x", 0.0))
                uy = float(u.get("y", 0.0))
                if math.hypot(x - ux, y - uy) <= xy_tol_mm:
                    # Average the positions into the existing cluster
                    n = u.get("_count", 1)
                    u["x"] = (ux * n + x) / (n + 1)
                    u["y"] = (uy * n + y) / (n + 1)
                    u["_count"] = n + 1
                    merged = True
                    break

            if not merged:
                # Start a new cluster (clone so we don't mutate original dict)
                new_d = dict(d)
                new_d["_count"] = 1
                unique.append(new_d)

        # Strip internal counter before returning
        for u in unique:
            u.pop("_count", None)

        return unique

    def _scan_side(self, side: str, duration: float = 10.0) -> List[Dict[str, Any]]:
        side = side.lower()
        self.status["scan_active"] = True
        self.status["scan_state"] = f"scanning_{side}"
        self.status["mode"] = "moving"
        start = time.monotonic()
        detections_all: List[Dict[str, Any]] = []

        # Open shield for scanning
        self._shield_open()
        # ---- CAMERA WARM-UP DELAY ----
        # Wait 1 second to let exposure / AWB settle,
        # discarding any frames during this time.
        warm_t0 = time.monotonic()
        while time.monotonic() - warm_t0 < 1.0:
            ok_w, frame_w = self.scan_cam.read_frame()
            time.sleep(0.01)
        # --------------------------------

        try:
            while time.monotonic() - start < duration:
                ok, frame = self.scan_cam.read_frame()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                # ultrasonic
                distance = self.ultra.latest_inches() if self.ultra else None

                # sensing
                result = self.sense.process(
                    frame,
                    distance_in=distance,
                    side=side,
                )

                # world-space detections
                detections_all.extend(result.detections)

                # update web status
                self._fps_frames += 1
                self.status["last_boxes"] = len(result.boxes)
                self.status["fps"] = self._fps_val
                self.status["label"] = result.label
                self.status["detection_center"] = result.detection_center

                # NEW: record this frame into the current run video (if active)
                if self.cam2 is not None:
                    ok2, frame2 = self.cam2.read_frame()
                    if ok2 and frame2 is not None:
                        self._record_frame(frame2)

                # save jpeg for stream
                jpeg = self.scan_cam.encode(result.frame)
                if jpeg:
                    with self._jpeg_lock:
                        self._latest_jpeg[:] = jpeg

        finally:
            # cleanup
            self._shield_close()
            self.status["scan_active"] = False
            self.status["scan_state"] = "idle"
            self.status["mode"] = "idle"

        # De-duplicate detections across all frames in this scan
        deduped = self._dedupe_detections(detections_all)

        print(
            f"[Scan] {side} side: collected {len(detections_all)} raw detections, "
            f"{len(deduped)} unique."
        )
        return deduped

    def _reorient_tool_if_needed(
        self,
        sock: Optional[socket.socket],
        desired_r: float,
        meta: Dict[str, Any],
        dry_run: bool = False,
    ) -> None:
        """
        Legacy hook for head reorientation. We no longer rotate J6 based on
        top/bottom rows; the tool is fixed at R=90 everywhere.

        This now just enforces that any commanded pose uses R=90 and logs a
        warning if not. It performs no motion.
        """
        if abs(desired_r - 90.0) > 1e-3:
            side = meta.get("side", "?")
            row = meta.get("row", "?")
            print(
                f"[Orientation] WARNING: requested R={desired_r:.3f} "
                f"on side={side}, row={row}; forcing R=90.0 in queue build."
            )

    def _execute_entry_with_crossing(
        self,
        sock: Optional[socket.socket],
        verb: str,
        coords: List[float],
        meta: Dict[str, Any],
        allow_spray: bool,
        dry_run: bool = False,
    ) -> None:
        """
        Execute a single verb/coords pair, inserting cross-beam excursions
        as needed. Head orientation is now fixed at J6=90; this function
        only handles side-crossing and near-surface safety.
        """
        start_pose = list(self._rob_pos_user)

        next_x = float(coords[0])
        current_x = float(self._rob_pos_user[0])

        crossing_r_to_l = (current_x > BEAM_MIDDLE_X) and (next_x < BEAM_MIDDLE_X)
        crossing_l_to_r = (current_x < BEAM_MIDDLE_X) and (next_x > BEAM_MIDDLE_X)
        target_inside_beam = (coords[2] < 60.0)

        sequence: List[Tuple[str, List[float]]] = []

        if crossing_r_to_l:
            # Exit beam on right, go to right homes, joint-move to left home, then drop back in if needed
            if self._rob_pos_user[2] < 60.0:
                r_exit = list(RIGHT_HOME_PARTIAL)
                if len(r_exit) == 6:
                    r_exit[5] = self._rob_pos_user[5]
                sequence.append(("movel", r_exit))

            sequence.append(("movel", list(RIGHT_HOME_PARTIAL)))
            sequence.append(("movel", list(RIGHT_HOME_REAL)))
            sequence.append(("movej", list(LEFT_HOME_REAL)))

            if target_inside_beam:
                l_partial = list(LEFT_HOME_PARTIAL)
                if len(l_partial) == 6:
                    l_partial[5] = coords[5]
                sequence.append(("movel", l_partial))

            sequence.append((verb, coords))

        elif crossing_l_to_r:
            # Exit beam on left, go to left homes, joint-move to right home, then drop back in if needed
            if self._rob_pos_user[2] < 60.0:
                l_exit = list(LEFT_HOME_PARTIAL)
                if len(l_exit) == 6:
                    l_exit[5] = self._rob_pos_user[5]
                sequence.append(("movel", l_exit))

            sequence.append(("movel", list(LEFT_HOME_PARTIAL)))
            sequence.append(("movel", list(LEFT_HOME_REAL)))
            sequence.append(("movej", list(RIGHT_HOME_REAL)))

            if target_inside_beam:
                r_partial = list(RIGHT_HOME_PARTIAL)
                if len(r_partial) == 6:
                    r_partial[5] = coords[5]
                sequence.append(("movel", r_partial))

            sequence.append((verb, coords))
        else:
            sequence.append((verb, coords))

        print(
            f"[Motion] last X={current_x:.3f}, next X={next_x:.3f}, "
            f"beam middle={BEAM_MIDDLE_X:.3f}"
        )
        for j, (v, c) in enumerate(sequence, start=1):
            preview = f"{v.upper()} " + " ".join(f"{float(n):.3f}" for n in c)
            print(f"  {j}. {preview}")

        for v, c in sequence:
            if not self._near_surface_safety_check(
                c,
                spray=bool(meta.get("spray", False)),
                context="orchestrator_execute",
            ):
                if meta.get("spray", False):
                    print("[Motion] Skipping unsafe spray move.")
                    return
                else:
                    print("[Motion] Skipping unsafe non-spray move.")
                    return

            if dry_run:
                print(f"[DRY RUN] Would send {v.upper()} to {', '.join(f'{float(n):.3f}' for n in c)}")
                self._rob_pos_user = list(c)
                continue

            ok = send_single_command(sock, v, c)
            if not ok:
                print("[Motion] Aborting sequence due to communication error.")
                raise RuntimeError("Communication error during motion.")
            self._rob_pos_user = list(c)
            save_last_pose(self._rob_pos_user)

        if allow_spray and meta.get("spray", False):
            if dry_run:
                print(
                    f"[DRY RUN] Would spray at target "
                    f"(side={meta.get('side')}, row={meta.get('row')})."
                )
            else:
                # Dynamic wait based on distance / 50 mm/s.
                delay = float(getattr(self, "_pending_spray_delay", 0.0))
                if delay > 0.0:
                    print(f"[Spray] Waiting {delay:.2f}s before spray for motion completion.")
                    time.sleep(delay)

                self.status["mode"] = "spraying"
                try:
                    # All timing (1s pre, spray_time, 1s post) is handled
                    # inside Servos.run_sequence() via lens_delay/spray_time.
                    self.servos.run_sequence()
                except Exception as e:
                    print(f"[Spray] Servo sequence error: {e}", file=sys.stderr)
                finally:
                    self.status["mode"] = "moving"
                    self._pending_spray_meta = None
                    self._pending_spray_delay = 0.0

    def _wait_for_ready_after_sequence(self, sock: socket.socket, timeout: float = 30.0) -> bool:
        """
        Block until we see a 'Ready for command' or until timeout.
        """
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

    def _execute_queue(self, queue: List[Dict[str, Any]], allow_spray: bool) -> bool:
        if not queue:
            print("[Motion] Empty queue; nothing to execute.")
            return True

        dry_run = bool(getattr(self.cfg, "dry_run", False))
        if dry_run:
            print("[Motion] DRY RUN is enabled; robot will NOT move or be contacted.")

        # Ensure the deferred-spray flag exists
        if not hasattr(self, "_pending_spray_meta"):
            self._pending_spray_meta = None  # type: ignore[attr-defined]

        sock: Optional[socket.socket] = None

        if not dry_run:
            attempts = 3
            delay_s = 2.0
            last_err: Optional[Exception] = None

            for attempt in range(1, attempts + 1):
                print(
                    f"[Motion] Connecting to robot at {ROBOT_IP}:{ROBOT_PORT} "
                    f"(attempt {attempt}/{attempts}) ..."
                )
                try:
                    sock = socket.create_connection(
                        (ROBOT_IP, ROBOT_PORT), timeout=SOCKET_TIMEOUT
                    )
                    sock.settimeout(SOCKET_TIMEOUT)
                    print("[Motion] Connected.")
                    break
                except OSError as e:
                    last_err = e
                    print(f"[Motion] Could not connect to robot: {e}")
                    if attempt < attempts:
                        print(f"[Motion] Retrying in {delay_s:.1f}s...")
                        time.sleep(delay_s)

            if sock is None:
                print("[Motion] Skipping this queue; robot pose left unchanged.")
                return False

        self.status["mode"] = "moving"

        try:
            for idx, entry in enumerate(queue):
                verb = entry.get("verb", "").strip().lower()
                coords = entry.get("coords", [])
                meta = entry.get("meta", {}) or {}

                if verb not in ("movel", "movej"):
                    print(f"[Motion] Invalid verb in entry #{idx}: {verb!r}")
                    return False
                if len(coords) != 6:
                    print(f"[Motion] Entry #{idx} must have 6 coords.")
                    return False

                print("\n======================================")
                print(f"[Motion] Queue entry #{idx + 1}")

                # If the previous entry was a spray target, perform the dwell + spray
                # sequence now, *before* we send any new robot motion.
                if (
                    not dry_run
                    and allow_spray
                    and self._pending_spray_meta is not None
                ):
                    prev = self._pending_spray_meta
                    side = prev.get("side", "?")
                    row = prev.get("row", "?")
                    print(
                        f"[Spray] Executing deferred spray for previous target "
                        f"(side={side}, row={row})."
                    )

                    # Dynamic wait based on distance / 50 mm/s.
                    delay = float(getattr(self, "_pending_spray_delay", 0.0))
                    if delay > 0.0:
                        print(f"[Spray] Waiting {delay:.2f}s before spray for motion completion.")
                        time.sleep(delay)

                    self.status["mode"] = "spraying"
                    try:
                        # All timing (1s pre, spray_time, 1s post) is handled
                        # inside Servos.run_sequence() via lens_delay/spray_time.
                        self.servos.run_sequence()
                    except Exception as e:
                        print(f"[Spray] Servo sequence error: {e}", file=sys.stderr)
                    finally:
                        self.status["mode"] = "moving"
                        self._pending_spray_meta = None
                        self._pending_spray_delay = 0.0

                # 1) Enforce orientation rule BEFORE any crossing or spraying:
                #    if R will change, go home on current side and do a MOVEJ
                #    that only changes joint 6.
                desired_r = float(coords[5])
                self._reorient_tool_if_needed(
                    sock=sock,
                    desired_r=desired_r,
                    meta=meta,
                    dry_run=dry_run,
                )

                # 2) Now execute with normal crossing logic and spraying rules.
                #    This may *mark* a spray target via self._pending_spray_meta,
                #    but the actual spray happens at the start of the *next* loop
                #    iteration in the block above.
                self._execute_entry_with_crossing(
                    sock, verb, coords, meta, allow_spray=allow_spray, dry_run=dry_run
                )

            if not dry_run and sock is not None:
                print("[Motion] Waiting for final 'Ready for command' (up to 30s)...")
                ready = self._wait_for_ready_after_sequence(sock, timeout=30.0)
                if not ready:
                    print(
                        "[Warning] Did not see 'Ready for command' within 30s after sequence.\n"
                        "          Using last commanded pose as rob_pos_user; verify robot status."
                    )
                else:
                    print("[Motion] Robot signaled 'Ready for command' again.")

                # Ensure last pose is persisted once more
                save_last_pose(self._rob_pos_user)

        except Exception as e:
            print(f"[Motion] Error while executing queue: {e}")
            self.status["mode"] = "error"
            return False
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
                print("[Motion] Disconnected from robot.")
            self.status["mode"] = "idle"
        return True

    def run(self) -> None:
        if self.ultra is not None:
            self.ultra.start()

        if self.cam2 is not None:
            self.cam2_thread = threading.Thread(target=self._cam2_loop, daemon=True)
            self.cam2_thread.start()

        self.website.start()

        # Ensure shield is closed at start of run
        self._shield_close()

        try:
            print(
                "\n[Start] Orchestrator ready.\n"
                "Sequence:\n"
                " 1) Move to LEFT_CAM and scan left side for rust\n"
                " 2) Build and present left-side targets\n"
                " 3) Execute left-side spray motions (if approved)\n"
                " 4) Move to RIGHT_CAM and repeat for right side\n"
                " 5) Return to LEFTCAM\n"
            )

            # Determine mode from config: default is auto (no user prompts)
            auto_mode = not self.cfg.run_interactive
            mode_label = "AUTO (no prompts)" if auto_mode else "INTERACTIVE (prompts enabled)"
            print(f"[Mode] Starting in {mode_label} based on config.run_interactive={self.cfg.run_interactive}")

            # === CONTINUOUS LOOP: repeat full scan/spray cycles ===
            while True:
                # LEFT side
                if not auto_mode:
                    ans = input(
                        "Press Enter to move robot to LEFT_CAM and start LEFT scan "
                        "(or type 'q' to quit): "
                    ).strip().lower()
                    if ans in ("q", "quit", "exit"):
                        print("[Start] Aborting before any motion.")
                        return
                else:
                    print("[Auto] Moving to LEFT_CAM and starting LEFT scan...")

                # --- NEW: start media capture for this run ---
                self._ensure_run_dir()
                self._capture_still("before")
                self._start_run_recording()
                # -------------------------------------------

                left_cam_coords = list(LEFT_CAM)
                if len(left_cam_coords) == 6:
                    left_cam_coords[5] = 90.0  # keep head at J6=90 for camera move too

                left_cam_entry = {
                    "verb": "movel",
                    "coords": left_cam_coords,
                    "meta": {"side": "left", "row": "cam", "source": "move_to_cam", "spray": False},
                }
                moved = self._execute_queue([left_cam_entry], allow_spray=False)
                if moved:
                    self._rob_pos_user = list(left_cam_coords)
                    save_last_pose(self._rob_pos_user)
                else:
                    print("[Start] Robot connection failed; continuing with vision-only test.")

                print("[Scan] Starting LEFT side scan (10 s)...")
                left_detections = self._scan_side("left", duration=5.0)

                left_queue: List[Dict[str, Any]] = []
                if left_detections:
                    left_queue = self._build_queue_for_side(left_detections, side="left")

                    print("\n[Left] Proposed targets (LEFT side):")
                    for i, entry in enumerate(left_queue, start=1):
                        coords = entry["coords"]
                        meta = entry.get("meta", {})
                        print(
                            f"  {i}. {entry['verb'].upper()} "
                            f"{' '.join(f'{float(c):.2f}' for c in coords)} "
                            f"meta={meta}"
                        )
                    try:
                        with open("command_queue_left.json", "w", encoding="utf-8") as f:
                            json.dump({"queue": left_queue}, f, indent=2)
                        print("[Left] Saved queue to command_queue_left.json")
                    except Exception as e:
                        print(f"[Left] Could not save command_queue_left.json: {e}", file=sys.stderr)

                    if not auto_mode:
                        ans = input(
                            "Execute LEFT side spray motions as listed above? [y/N]: "
                        ).strip().lower()
                        execute_left = ans in ("y", "yes")
                    else:
                        print("[Auto] Executing LEFT side spray motions.")
                        execute_left = True

                    if execute_left:
                        _ = self._execute_queue(left_queue, allow_spray=True)
                    else:
                        print("[Left] Skipping LEFT side motions.")
                else:
                    print("[Left] No rust targets detected on left side.")

                # RIGHT side
                if not auto_mode:
                    ans = input(
                        "\nPress Enter to move robot to RIGHT_CAM and start RIGHT scan "
                        "(or type 'q' to skip right side): "
                    ).strip().lower()
                    skip_right = ans in ("q", "quit", "exit")
                else:
                    print("[Auto] Moving to RIGHT_CAM and starting RIGHT scan...")
                    skip_right = False

                if not skip_right:
                    right_cam_coords = list(RIGHT_CAM)
                    if len(right_cam_coords) == 6:
                        right_cam_coords[5] = 90.0
                    right_cam_entry = {
                        "verb": "movel",
                        "coords": right_cam_coords,
                        "meta": {
                            "side": "right",
                            "row": "cam",
                            "source": "move_to_cam",
                            "spray": False,
                        },
                    }

                    moved_r = self._execute_queue([right_cam_entry], allow_spray=False)

                    if not moved_r:
                        # *** IMPORTANT: bail out of right side completely ***
                        print("[Right] Robot move to RIGHT_CAM failed; skipping RIGHT scan and motions.")
                        right_detections: List[Dict[str, Any]] = []
                    else:
                        self._rob_pos_user = list(right_cam_coords)
                        save_last_pose(self._rob_pos_user)

                        print("[Scan] Starting RIGHT side scan (10 s)...")
                        right_detections = self._scan_side("right", duration=5.0)

                    right_queue: List[Dict[str, Any]] = []
                    if right_detections:
                        right_queue = self._build_queue_for_side(right_detections, side="right")

                        print("\n[Right] Proposed targets (RIGHT side):")
                        for i, entry in enumerate(right_queue, start=1):
                            coords = entry["coords"]
                            meta = entry.get("meta", {})
                            print(
                                f"  {i}. {entry['verb'].upper()} "
                                f"{' '.join(f'{float(c):.2f}' for c in coords)} "
                                f"meta={meta}"
                            )
                        try:
                            with open("command_queue_right.json", "w", encoding="utf-8") as f:
                                json.dump({"queue": right_queue}, f, indent=2)
                            print("[Right] Saved queue to command_queue_right.json")
                        except Exception as e:
                            print(f"[Right] Could not save command_queue_right.json: {e}", file=sys.stderr)

                        if not auto_mode:
                            ans = input(
                                "Execute RIGHT side spray motions as listed above? [y/N]: "
                            ).strip().lower()
                            execute_right = ans in ("y", "yes")
                        else:
                            print("[Auto] Executing RIGHT side spray motions.")
                            execute_right = True

                        if execute_right:
                            _ = self._execute_queue(right_queue, allow_spray=True)
                        else:
                            print("[Right] Skipping RIGHT side motions.")
                    else:
                        print("[Right] No rust targets detected on right side.")
                else:
                    print("[Right] Skipping RIGHT side entirely by user request.")

                # Final return
                print("\n[Return] Moving back to LEFT_CAM before next cycle...")
                final_coords = list(LEFT_CAM)
                if len(final_coords) == 6:
                    final_coords[5] = 90.0
                final_queue: List[Dict[str, Any]] = [
                    {
                        "verb": "movel",
                        "coords": final_coords,
                        "meta": {"side": "left", "row": "cam", "source": "final", "spray": False},
                    },
                ]
                _ = self._execute_queue(final_queue, allow_spray=False)
                print("\n[Done] Full scan & spray sequence complete. Starting next cycle...\n")

                # --- NEW: end-of-run media handling ---
                self._capture_still("after")
                self._stop_run_recording()
                self._upload_run_media()
                # -------------------------------------

        except KeyboardInterrupt:
            print("\n[Exit] Ctrl+C")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._cam2_running = False
        if self.cam2_thread is not None and self.cam2_thread.is_alive():
            try:
                self.cam2_thread.join(timeout=1.0)
            except Exception:
                pass

        try:
            self.cam.release()
        except Exception:
            pass
        if self.cam2 is not None:
            try:
                self.cam2.release()
            except Exception:
                pass

        try:
            self.servos.park()
        except Exception:
            pass

        if self.ultra is not None:
            try:
                self.ultra.stop()
            except Exception:
                pass

        try:
            self.website.stop()
        except Exception:
            pass

        print("[Exit] Clean shutdown.")
