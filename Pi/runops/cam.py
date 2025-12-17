import sys
from typing import Optional

import cv2
import numpy as np
import subprocess

from config import Config


class JpegEncoder:
    def __init__(self, cfg: Config):
        self.backend = "opencv"
        self.jq = cfg.opencv_quality
        self.jpeg = None
        self.stream_downscale_width = cfg.stream_downscale_width

        try:
            from turbojpeg import TurboJPEG, TJPF_BGR
            self.jpeg = TurboJPEG()
            self.TJPF_BGR = TJPF_BGR
            self.backend = "turbojpeg"
            self.jq = cfg.turbojpeg_quality
            print("[Encoder] Using TurboJPEG")
        except Exception as e:
            print(f"[Encoder] TurboJPEG unavailable, falling back to OpenCV: {e}",
                  file=sys.stderr)

    def encode(self, frame_bgr: np.ndarray) -> bytes:
        img = frame_bgr

        # Optional downscale
        if self.stream_downscale_width and self.stream_downscale_width > 0:
            h, w = frame_bgr.shape[:2]
            if w > self.stream_downscale_width:
                new_w = self.stream_downscale_width
                new_h = int(h * (new_w / float(w)))
                img = cv2.resize(frame_bgr, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)

        if self.backend == "turbojpeg" and self.jpeg is not None:
            return self.jpeg.encode(img, quality=self.jq, pixel_format=self.TJPF_BGR)
        else:
            ok, buf = cv2.imencode(".jpg", img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jq)])
            return buf.tobytes() if ok else b""


class Cam:
    def __init__(self, cfg: Config, cam_index: Optional[int] = None):
        self.cfg = cfg
        # If cam_index is given, use it; otherwise use cfg.cam_index
        self.cam_index = cfg.cam_index if cam_index is None else cam_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.encoder = JpegEncoder(cfg)
        self._open_camera()

    @property
    def encoder_backend(self) -> str:
        return self.encoder.backend

    def _open_camera(self):
        backend = cv2.CAP_V4L2
        cap = cv2.VideoCapture(self.cam_index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_h)
        if self.cfg.use_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, self.cfg.req_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.cam_index}.")

        # Prime buffer
        for _ in range(5):
            cap.read()

        try:
            print(f"[Camera {self.cam_index}] Negotiated FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}")
        except Exception:
            pass

        self.cap = cap

        # Apply manual v4l2 controls only for the primary camera, if enabled
        try:
            self._apply_v4l2_controls_if_primary()
        except Exception as e:
            print(f"[Camera {self.cam_index}] v4l2 control setup failed: {e}", file=sys.stderr)

    def _apply_v4l2_controls_if_primary(self) -> None:
        """
        Apply focus / exposure / white-balance using v4l2-ctl, but ONLY if:
          - this camera is the primary cam (cam_index == cfg.cam_index), and
          - cfg.enable_camera_v4l2_control is True.

        All values are taken from config so you can tune them there.
        """
        # Only touch the primary camera
        if self.cam_index != getattr(self.cfg, "cam_index", self.cam_index):
            return

        if not bool(getattr(self.cfg, "enable_camera_v4l2_control", False)):
            print(f"[Camera {self.cam_index}] v4l2 control disabled by config.")
            return

        # Which /dev/video* should we control?
        device_path = getattr(
            self.cfg,
            "cam_control_device",
            f"/dev/video{self.cam_index}",
        )

        # Pull settings from config with safe defaults
        focus_auto = int(getattr(self.cfg, "manual_focus_auto", 0))  # 0 = manual, 1 = auto (driver-specific)
        focus_val = int(getattr(self.cfg, "manual_focus_value", 50))

        # For this webcam: auto_exposure=1 is usually "manual" mode
        auto_exposure_mode = int(getattr(self.cfg, "manual_auto_exposure_mode", 1))
        exposure_val = int(getattr(self.cfg, "manual_exposure_value", 800))

        wb_auto = int(getattr(self.cfg, "manual_white_balance_auto", 0))
        wb_temp = int(getattr(self.cfg, "manual_wb_temperature", 4600))

        gain_val = int(getattr(self.cfg, "manual_gain_value", 120))

        def _run_ctrl(name: str, ctrl: str):
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", device_path, f"--set-ctrl={ctrl}"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                print(f"[Camera {self.cam_index}] Failed to set {name} on {device_path}", file=sys.stderr)

        print(f"[Camera {self.cam_index}] Applying manual v4l2 controls if supported on {device_path}")

        # Focus
        _run_ctrl("focus_automatic_continuous", f"focus_automatic_continuous={focus_auto}")
        if focus_auto == 0:
            _run_ctrl("focus_absolute", f"focus_absolute={focus_val}")

        # Exposure
        _run_ctrl("auto_exposure", f"auto_exposure={auto_exposure_mode}")
        if auto_exposure_mode == 1:
            _run_ctrl("exposure_time_absolute", f"exposure_time_absolute={exposure_val}")

        # White balance
        _run_ctrl("white_balance_automatic", f"white_balance_automatic={wb_auto}")
        if wb_auto == 0:
            _run_ctrl("white_balance_temperature", f"white_balance_temperature={wb_temp}")

        # Gain (helps brighten dark image)
        _run_ctrl("gain", f"gain={gain_val}")

        print(
            f"[Camera {self.cam_index}] Final camera state target:"
            f" focus_auto={focus_auto}, focus={focus_val},"
            f" exp_mode={auto_exposure_mode}, exp={exposure_val},"
            f" wb_auto={wb_auto}, wb={wb_temp}, gain={gain_val}"
        )

    def read_frame(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def encode(self, frame):
        return self.encoder.encode(frame)

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
