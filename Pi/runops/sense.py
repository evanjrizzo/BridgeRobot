
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import math
import json
from pathlib import Path

from config import Config

class SenseResult:
    """Result object returned by Sense.process to the rest of the system."""

    def __init__(self, frame_bgr, boxes: List[Tuple[int, int, int, int]], detection_center: Optional[Dict[str, float]], label: str, detections: List[Dict[str, float]]):
        self.frame = frame_bgr
        self.boxes = boxes
        self.detection_center = detection_center
        self.label = label
        # All homography-based candidate targets
        self.detections = detections

    @property
    def has_rust(self) -> bool:
        return bool(self.boxes)

class Sense:
    """
    Rust-spot detector for debugging homography and user-frame mapping.

    Design choices:
      * Pixel-space filtering is done ONLY via a simple beam ROI that masks
        out everything outside the beam zone BEFORE contour detection.
      * The main geometric filter is user-frame bounds on (X,Y) after
        homography.
      * X,Y we output are taken directly from homography. No extra transforms.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.verbose = bool(getattr(cfg, "sense_verbose", False))

        # Base HSV window for the rust color (H fixed, S/V tunable).
        self.base_lower = np.array([8, 80, 70], dtype=np.uint8)
        self.base_upper = np.array([25, 255, 255], dtype=np.uint8)

        # Active thresholds. auto_calibrate() may update these.
        self.lower = self.base_lower.copy()
        self.upper = self.base_upper.copy()

        # Homography matrices (image -> user-frame X,Y).
        self.H_left: Optional[np.ndarray] = None
        self.H_right: Optional[np.ndarray] = None
        self._load_homographies()

        # Inverse homographies (user-frame X,Y -> image pixels) for grid drawing.
        self.Hinv_left: Optional[np.ndarray] = None
        self.Hinv_right: Optional[np.ndarray] = None
        if self.H_left is not None:
            try:
                self.Hinv_left = np.linalg.inv(self.H_left)
            except np.linalg.LinAlgError:
                self.Hinv_left = None
        if self.H_right is not None:
            try:
                self.Hinv_right = np.linalg.inv(self.H_right)
            except np.linalg.LinAlgError:
                self.Hinv_right = None

        # Beam middle in user-frame coordinates (mm). Used only for
        # classification (top/bottom, left/right), not to change X/Y.
        self.beam_middle_x = getattr(cfg, "beam_middle_x", None)
        self.beam_middle_y = getattr(cfg, "beam_middle_y", None)

        # Optional inversion of head rotation logic. Only affects R suggestion.
        self.invert_head_rotation = getattr(cfg, "invert_head_rotation", False)

        # Debug options for drawing world grid overlays.
        # If these keys don't exist in config, defaults are used.
        self.debug_grid = bool(getattr(cfg, "debug_grid", False))
        self.grid_step_x_mm = getattr(cfg, "debug_grid_step_x_mm", 100.0)
        self.grid_step_y_mm = getattr(cfg, "debug_grid_step_y_mm", 100.0)

        # Pixel-space beam ROIs.
        # Each is [min_x, max_x, min_y, max_y] in PIXELS.
        # These are the exclusion zones you derived from your AprilTags.
        self.beam_roi_global = getattr(cfg, "beam_roi_global", None)
        self.beam_roi_left = getattr(cfg, "beam_roi_left", None)
        self.beam_roi_right = getattr(cfg, "beam_roi_right", None)
        if self.verbose:
            print(
                "[Sense] ROIs:",
                "global=", self.beam_roi_global,
                "left=", self.beam_roi_left,
                "right=", self.beam_roi_right)
        # Global X/Y bounds (fallbacks) in user frame.
        self.x_min_global = getattr(cfg, "target_x_min", None)
        self.x_max_global = getattr(cfg, "target_x_max", None)
        self.y_min_global = getattr(cfg, "target_y_min", None)
        self.y_max_global = getattr(cfg, "target_y_max", None)

        # Side-specific X/Y bounds for LEFT.
        self.x_min_left = getattr(cfg, "target_x_min_left", self.x_min_global)
        self.x_max_left = getattr(cfg, "target_x_max_left", self.x_max_global)
        self.y_min_left = getattr(cfg, "target_y_min_left", self.y_min_global)
        self.y_max_left = getattr(cfg, "target_y_max_left", self.y_max_global)

        # Side-specific X/Y bounds for RIGHT.
        self.x_min_right = getattr(cfg, "target_x_min_right", self.x_min_global)
        self.x_max_right = getattr(cfg, "target_x_max_right", self.x_max_global)
        self.y_min_right = getattr(cfg, "target_y_min_right", self.y_min_global)
        self.y_max_right = getattr(cfg, "target_y_max_right", self.y_max_global)

        # Grid spacing in user frame (mm) for visualization grid.
        self.grid_step_x_mm = getattr(cfg, "grid_step_x_mm", 100.0)
        self.grid_step_y_mm = getattr(cfg, "grid_step_y_mm", 50.0)

    # ------------------------------------------------------------------
    # Homography helpers
    # ------------------------------------------------------------------

    def _load_homographies(self) -> None:
        """Load homography_left.json / homography_right.json if present."""
        base_dir = Path(__file__).resolve().parent

        left_path = base_dir / "homography_left.json"
        right_path = base_dir / "homography_right.json"

        def _load(path: Path) -> Optional[np.ndarray]:
            if not path.exists():
                print(f"[Sense] Homography file not found: {path}")
                return None
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                H = np.asarray(data["H"], dtype=np.float64)
                if H.shape != (3, 3):
                    print(f"[Sense] Bad homography shape in {path}: {H.shape}")
                    return None
                return H
            except Exception as e:
                print(f"[Sense] Failed to load homography from {path}: {e}")
                return None

        self.H_left = _load(left_path)
        self.H_right = _load(right_path)

        if self.H_left is not None:
            print("[Sense] Loaded homography_left.json")
        else:
            print("[Sense] WARNING: homography_left.json not found/invalid.")

        if self.H_right is not None:
            print("[Sense] Loaded homography_right.json")
        else:
            print("[Sense] WARNING: homography_right.json not found/invalid.")

    def _pixel_to_world_xy(
        self, u: float, v: float, side: Optional[str]
    ) -> Optional[Tuple[float, float]]:
        """
        Convert (u,v) pixels to user-frame (X,Y) using the homography for the
        given side. This is the ONLY transform applied to produce X,Y.
        """
        if side == "left":
            H = self.H_left
        elif side == "right":
            H = self.H_right
        else:
            H = None

        if H is None:
            return None

        pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
        try:
            world = cv2.perspectiveTransform(pt, H).reshape(2)
            X, Y = float(world[0]), float(world[1])
            return X, Y
        except Exception as e:
            print(f"[Sense] Homography transform failed for side={side}: {e}")
            return None

    def _world_to_pixel(
        self,
        X: float,
        Y: float,
        side: Optional[str],
        frame_shape: Tuple[int, int, int],
    ) -> Optional[Tuple[int, int]]:
        """
        Map user-frame (X,Y) back to pixel coordinates (u,v) using the
        inverse homography for the given side. Returns None if mapping fails
        or lands outside the image bounds.
        """
        if side == "left":
            Hinv = self.Hinv_left
        elif side == "right":
            Hinv = self.Hinv_right
        else:
            # Fallback if side is None: prefer left, then right.
            Hinv = self.Hinv_left or self.Hinv_right

        if Hinv is None:
            return None

        h, w = frame_shape[:2]

        vec = np.array([X, Y, 1.0], dtype=np.float64)
        uvw = Hinv @ vec
        if abs(uvw[2]) < 1e-9:
            return None

        u = uvw[0] / uvw[2]
        v = uvw[1] / uvw[2]

        u_i = int(round(u))
        v_i = int(round(v))

        if u_i < 0 or u_i >= w or v_i < 0 or v_i >= h:
            return None

        return (u_i, v_i)

    # ------------------------------------------------------------------
    # Bounds checking in user coordinates
    # ------------------------------------------------------------------

    def _within_bounds(
        self,
        X: float,
        Y: float,
        side: Optional[str] = None,
    ) -> bool:
        """
        Check whether user-frame (X, Y) lies inside the configured bounds.

        Bounds are taken from Config:
          - target_x_min / target_x_max / target_y_min / target_y_max
          - target_x_min_left / target_x_max_left / target_y_min_left / target_y_max_left
          - target_x_min_right / target_x_max_right / target_y_min_right / target_y_max_right

        If a bound is not set (None), it's ignored.
        """

        cfg = self.cfg

        # Global defaults from config (may be None).
        x_min_global = getattr(cfg, "target_x_min", None)
        x_max_global = getattr(cfg, "target_x_max", None)
        y_min_global = getattr(cfg, "target_y_min", None)
        y_max_global = getattr(cfg, "target_y_max", None)

        # Side-specific overrides.
        if side == "left":
            x_min = getattr(cfg, "target_x_min_left", x_min_global)
            x_max = getattr(cfg, "target_x_max_left", x_max_global)
            y_min = getattr(cfg, "target_y_min_left", y_min_global)
            y_max = getattr(cfg, "target_y_max_left", y_max_global)
        elif side == "right":
            x_min = getattr(cfg, "target_x_min_right", x_min_global)
            x_max = getattr(cfg, "target_x_max_right", x_max_global)
            y_min = getattr(cfg, "target_y_min_right", y_min_global)
            y_max = getattr(cfg, "target_y_max_right", y_max_global)
        else:
            # Fallback: just use global.
            x_min, x_max = x_min_global, x_max_global
            y_min, y_max = y_min_global, y_max_global

        # Now apply the checks. Any None bound is ignored.
        if x_min is not None and X < x_min:
            return False
        if x_max is not None and X > x_max:
            return False
        if y_min is not None and Y < y_min:
            return False
        if y_max is not None and Y > y_max:
            return False

        return True

    # ------------------------------------------------------------------
    # Pixel-space ROI (simple exclusion)
    # ------------------------------------------------------------------

    def _apply_beam_roi(self, mask: np.ndarray, side: Optional[str]) -> np.ndarray:
        """
        Zero out everything outside the configured beam ROI in PIXEL space.

        ROI format: [min_x, max_x, min_y, max_y] in pixels.
        If no ROI is configured for this side (or globally), the mask is returned unchanged.
        """
        roi = None
        if side == "left":
            roi = self.beam_roi_left or self.beam_roi_global
        elif side == "right":
            roi = self.beam_roi_right or self.beam_roi_global
        else:
            roi = self.beam_roi_global

        if roi is None:
            return mask

        h, w = mask.shape[:2]
        min_x, max_x, min_y, max_y = roi

        # Clamp and handle None gracefully.
        x0 = 0 if min_x is None else max(0, int(round(min_x)))
        x1 = w if max_x is None else min(w, int(round(max_x)))
        y0 = 0 if min_y is None else max(0, int(round(min_y)))
        y1 = h if max_y is None else min(h, int(round(max_y)))

        if x0 >= x1 or y0 >= y1:
            # Bad ROI; just blank everything so nothing is ever detected.
            return np.zeros_like(mask)

        roi_mask = np.zeros_like(mask)
        roi_mask[y0:y1, x0:x1] = 255

        return cv2.bitwise_and(mask, roi_mask)

    # ------------------------------------------------------------------
    # World grid overlay
    # ------------------------------------------------------------------

    def _draw_world_grid(self, frame_bgr: np.ndarray, side: Optional[str]) -> None:
        """
        Overlay a grid of user-frame coordinates on the image, based on
        the side-specific X/Y bounds and homography.

        This is visualization-only; it does not affect detection.
        """
        if side == "left":
            x_min = self.x_min_left
            x_max = self.x_max_left
            y_min = self.y_min_left
            y_max = self.y_max_left
        elif side == "right":
            x_min = self.x_min_right
            x_max = self.x_max_right
            y_min = self.y_min_right
            y_max = self.y_max_right
        else:
            x_min = self.x_min_global
            x_max = self.x_max_global
            y_min = self.y_min_global
            y_max = self.y_max_global

        # Need valid bounds and inverse homography
        if None in (x_min, x_max, y_min, y_max):
            return

        if side == "left" and self.Hinv_left is None:
            return
        if side == "right" and self.Hinv_right is None:
            return
        if side not in ("left", "right") and (self.Hinv_left is None and self.Hinv_right is None):
            return

        h, w = frame_bgr.shape[:2]

        # X and Y ticks in user frame
        try:
            step_x = float(self.grid_step_x_mm)
            step_y = float(self.grid_step_y_mm)
        except Exception:
            step_x = 100.0
            step_y = 50.0

        if step_x <= 0 or step_y <= 0:
            return

        xs = np.arange(x_min, x_max + 1e-3, step_x, dtype=float)
        ys = np.arange(y_min, y_max + 1e-3, step_y, dtype=float)

        # Draw vertical lines at each X
        for X in xs:
            p1 = self._world_to_pixel(X, y_min, side, frame_bgr.shape)
            p2 = self._world_to_pixel(X, y_max, side, frame_bgr.shape)
            if p1 is None or p2 is None:
                continue
            cv2.line(frame_bgr, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
            label_pos = (p1[0] + 2, max(0, p1[1] - 4))
            cv2.putText(
                frame_bgr,
                f"X={int(round(X))}",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw horizontal lines at each Y
        for Y in ys:
            p1 = self._world_to_pixel(x_min, Y, side, frame_bgr.shape)
            p2 = self._world_to_pixel(x_max, Y, side, frame_bgr.shape)
            if p1 is None or p2 is None:
                continue
            cv2.line(frame_bgr, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
            label_pos = (max(0, p1[0] - 60), p1[1] + 12)
            cv2.putText(
                frame_bgr,
                f"Y={int(round(Y))}",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # Simple color sweep calibration
    # ------------------------------------------------------------------

    def _score_mask(self, mask: np.ndarray) -> float:
        """Score a mask by the area of its largest contour."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        return float(cv2.contourArea(largest))

    def auto_calibrate(self, frame_bgr: np.ndarray) -> None:
        """
        Coarse sweep over S/V around the base HSV hue to pick a better mask
        for the current lighting.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        best_score = -1.0
        best_lower = self.lower
        best_upper = self.upper

        base_l = self.base_lower.astype(int)
        base_u = self.base_upper.astype(int)

        s_shifts = [-40, -20, 0, 20, 40]
        v_shifts = [-40, -20, 0, 20, 40]

        for ds in s_shifts:
            for dv in v_shifts:
                l_s = int(np.clip(base_l[1] + ds, 0, 255))
                l_v = int(np.clip(base_l[2] + dv, 0, 255))
                u_s = int(np.clip(base_u[1] + ds, 0, 255))
                u_v = int(np.clip(base_u[2] + dv, 0, 255))

                lower = np.array([base_l[0], l_s, l_v], dtype=np.uint8)
                upper = np.array([base_u[0], u_s, u_v], dtype=np.uint8)

                mask = cv2.inRange(hsv, lower, upper)
                score = self._score_mask(mask)

                if score > best_score:
                    best_score = score
                    best_lower = lower
                    best_upper = upper

        self.lower = best_lower
        self.upper = best_upper
        if self.verbose:
            print(
                f"[Sense] Calibrated HSV thresholds: "
                f"lower={self.lower}, upper={self.upper}, score={best_score:.1f}"
            )

    # ------------------------------------------------------------------
    # Main detection entry point
    # ------------------------------------------------------------------
    def process(
        self,
        frame_bgr,
        distance_in: Optional[float] = None,
        side: Optional[str] = None,
        tag_list=None,  # accepted for compatibility, currently unused
    ) -> SenseResult:
        """
        Detect rust spots in a frame and generate world-space subtargets
        spaced roughly max_target_segment_mm apart *along the interior of
        each blob* (using a simple skeleton-style approach).

        side:
            "left" or "right" so we can:
              - choose the correct homography
              - apply side-specific X/Y bounds in user frame

        Pixel-space exclusion is done via the beam ROI; all final
        geometric filtering is done in user-frame (X, Y) by _within_bounds().
        """
        # Basic image sizes (kept in case you need them later)
        H_img, W_img = frame_bgr.shape[:2]
        cy_img = H_img / 2.0

        # HSV threshold for the current frame.
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # HARD CUT: keep only the beam zone in pixel space.
        mask = self._apply_beam_roi(mask, side=side)

        # Basic cleanup: small morphological open + dilate + blur.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # Find external contours.
        contours_info = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = contours_info[1] if len(contours_info) == 3 else contours_info[0]

        # ---------------------------------------------
        # BUILD WORLD-SPACE DETECTIONS FROM BLOBS
        # ---------------------------------------------
        frame_h, frame_w = frame_bgr.shape[:2]

        boxes: List[Tuple[int, int, int, int]] = []
        candidates: List[Dict[str, float]] = []

        # Desired world spacing between interior subtargets (mm).
        max_seg_len_mm = float(getattr(self.cfg, "max_target_segment_mm", 40.0))

        for c in cnts:
            # Reject small contours
            area = cv2.contourArea(c)
            if area < self.cfg.area_min:
                continue

            # Pixel-space bounding box (for visualization and metadata)
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))

            # Center of bounding box in pixels (used for some fallbacks)
            center_x = x + w // 2
            center_y = y + h // 2

            # Compute world center (for possible fallback / bounds checks)
            world_center = self._pixel_to_world_xy(center_x, center_y, side=side)
            if world_center is None:
                # Homography failed for this region; skip.
                continue

            # ----------------------------------------------------------
            # Build a filled mask for just this contour.
            # ----------------------------------------------------------
            comp_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(comp_mask, [c], -1, 255, thickness=-1)

            # Distance transform inside this blob.
            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)

            # Local maxima of distance transform ≈ skeleton centerline.
            # Trick: pixels where dist == dilated(dist) and comp_mask > 0.
            dist_dil = cv2.dilate(dist, None)
            local_max = (dist == dist_dil) & (comp_mask > 0)
            skeleton_mask = np.zeros_like(comp_mask, dtype=np.uint8)
            skeleton_mask[local_max] = 255

            # If skeleton is empty (very thin blob / weird geometry), fall
            # back to using the entire filled component.
            if not np.any(skeleton_mask):
                skeleton_mask = comp_mask.copy()

            # Pixel coordinates of skeleton (or fallback) points.
            ys, xs = np.where(skeleton_mask > 0)
            if xs.size == 0:
                continue

            # Optionally thin out raw skeleton pixels in pixel-space to
            # keep the mapping cost manageable.
            # Here we just stride them; world decimation comes next.
            stride = max(1, int(round(len(xs) / 1000)))  # cap ~1000 samples
            xs = xs[::stride]
            ys = ys[::stride]

            # Map skeleton pixels to world and keep only in-bounds points.
            world_pts: List[Tuple[float, float, int, int]] = []
            for u_px, v_px in zip(xs, ys):
                res = self._pixel_to_world_xy(int(u_px), int(v_px), side=side)
                if res is None:
                    continue
                wx, wy = res
                if not self._within_bounds(wx, wy, side=side):
                    continue
                world_pts.append((wx, wy, int(u_px), int(v_px)))

            if not world_pts:
                # If nothing mapped cleanly, fall back to single-center target.
                wx_center, wy_center = world_center
                if not self._within_bounds(wx_center, wy_center, side=side):
                    continue

                if self.beam_middle_y is not None:
                    is_top_row = wy_center > self.beam_middle_y
                else:
                    is_top_row = center_y > frame_h / 2

                R_suggest = 0.0 if is_top_row else 180.0

                half = None
                if self.beam_middle_x is not None:
                    half = "left" if wx_center < self.beam_middle_x else "right"

                candidates.append(
                    {
                        "x": float(wx_center),
                        "y": float(wy_center),
                        "is_top": bool(is_top_row),
                        "half": half,
                        "R": float(R_suggest),
                        "u": float(center_x),
                        "v": float(center_y),
                        "box": (int(x), int(y), int(w), int(h)),
                    }
                )
                continue

            # ----------------------------------------------------------
            # Sort skeleton points in world-space and place subtargets
            # along that interior path at ~max_seg_len_mm spacing.
            # ----------------------------------------------------------
            # Sort by world X (works well for mostly horizontal beam).
            world_pts.sort(key=lambda p: p[0])

            # Start at the first skeleton point.
            prev_x_w, prev_y_w, prev_u, prev_v = world_pts[0]
            wx0, wy0 = prev_x_w, prev_y_w

            # We'll always keep at least one target per blob.
            # Defer adding until after we compute top/bottom, etc., so we
            # can share code with the loop.
            path_points: List[Tuple[float, float, int, int]] = []
            path_points.append((prev_x_w, prev_y_w, prev_u, prev_v))

            acc_dist = 0.0
            for wx, wy, u_px, v_px in world_pts[1:]:
                step = math.hypot(wx - prev_x_w, wy - prev_y_w)
                acc_dist += step
                prev_x_w, prev_y_w = wx, wy

                if acc_dist >= max_seg_len_mm:
                    path_points.append((wx, wy, u_px, v_px))
                    acc_dist = 0.0

            # Now convert each kept interior point into a candidate.
            for wx, wy, u_px, v_px in path_points:
                # Determine top vs bottom row in user frame if possible.
                if self.beam_middle_y is not None:
                    is_top_row = wy > self.beam_middle_y
                else:
                    # fallback using image height
                    is_top_row = v_px > frame_h / 2

                # Determine R angle (0° for bottom, 180° for top)
                R_suggest = 0.0 if is_top_row else 180.0

                # Determine left/right half using X
                half = None
                if self.beam_middle_x is not None:
                    half = "left" if wx < self.beam_middle_x else "right"

                # Use the original pixel location for visualization.
                u_vis, v_vis = int(u_px), int(v_px)

                candidates.append(
                    {
                        "x": float(wx),
                        "y": float(wy),
                        "is_top": bool(is_top_row),
                        "half": half,
                        "R": float(R_suggest),
                        "u": float(u_vis),
                        "v": float(v_vis),
                        "box": (int(x), int(y), int(w), int(h)),
                    }
                )

        # ------------------------------------------------------
        # THIN CANDIDATES: enforce ~one point every max_seg_len_mm
        # ------------------------------------------------------
        if candidates:
            # Sort for deterministic behavior (left-to-right, then top-to-bottom)
            candidates.sort(key=lambda d: (d["x"], d["y"]))

            thinned: List[Dict[str, float]] = []
            # Use a little less than max_seg_len_mm so we don't end up *over* 40–50 mm
            min_dist = max_seg_len_mm * 0.8

            for det in candidates:
                if not thinned:
                    thinned.append(det)
                    continue

                x = det["x"]
                y = det["y"]

                keep = True
                for kept in thinned:
                    if math.hypot(x - kept["x"], y - kept["y"]) < min_dist:
                        keep = False
                        break

                if keep:
                    thinned.append(det)

            candidates = thinned

        detection_center: Optional[Dict[str, float]] = None


        if candidates:
            # Keep the first candidate as "primary" (for compatibility).
            detection_center = candidates[0]

            if self.verbose:
                print(
                    f"[Sense] FINAL detection_center side={side} "
                    f"X={detection_center['x']:.2f} "
                    f"Y={detection_center['y']:.2f} "
                    f"u={detection_center['u']:.1f} "
                    f"v={detection_center['v']:.1f}"
                )

            # Draw all candidate targets on the frame.
            for det in candidates:
                cx = int(det["u"])
                cy = int(det["v"])

                # Make the primary target a bit more obvious.
                radius = 6 if det is detection_center else 4
                cv2.circle(frame_bgr, (cx, cy), radius, (0, 0, 255), -1)

                ux = det["x"]
                uy = det["y"]
                text = f"({int(ux)}, {int(uy)})"
                tx = max(0, cx + 5)
                ty = max(15, cy - 8)
                cv2.putText(
                    frame_bgr,
                    text,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # Draw all in-bounds boxes for visualization.
        label = "RUST DETECTED" if boxes else "NO RUST DETECTED"
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(frame_bgr, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)

        if self.verbose:
            print(f"[Sense] {label}: {len(candidates)} subtargets found (side={side})")

        return SenseResult(frame_bgr, boxes, detection_center, label, candidates)
