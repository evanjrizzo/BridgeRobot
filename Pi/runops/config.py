# config.py
import json
import os
from dataclasses import dataclass
from typing import Tuple, Any, Dict, Optional, List

CONFIG_FILE = "config.json"


@dataclass
class Config:
    run_interactive: bool = False

    # Spray timing (seconds)
    pre_spray_delay_s: float = 1.0     # before opening valve
    spray_duration_s: float = 1.0      # valve open time
    post_spray_delay_s: float = 1.0    # after closing valve

    # Shield servo timing (seconds)
    shield_move_time_s: float = 0.4    # time for SG90 to reach position

    # Network / backend
    host: str = "0.0.0.0"
    port: int = 8080

    # Camera
    # Use the secondary camera (cam2) for scanning/targeting instead of primary
    use_cam2_for_scan = False
    enable_camera_v4l2_control = False

    # Make sure indices reflect your USB cams:
    cam_index = 0      # primary cam object (old primary or whatever)
    cam2_index = 0     # secondary cam object (now physically at main spot)

    # Per-feature auto toggles for cam0
    enable_cam0_autofocus = True        # True = use camera AF, False = manual + sweep
    enable_cam0_auto_exposure = False    # True = camera AE, False = manual + sweep
    enable_cam0_auto_wb = True     # True = camera AWB, False = manual + sweep

    # Manual camera controls (UVC / v4l2)
    manual_focus: int = 50          # 0–1023 for your camera
    manual_exposure: int = 1000       # 1–10000 (exposure_time_absolute)
    manual_wb_temp: int = 4600       # 2800–6500 if you want fixed WB

    enable_color_auto_calibration = False

    # Turn sweeps on/off
    run_focus_sweep_on_start = False
    run_exposure_sweep_on_start = False
    run_wb_sweep_on_start = False

    # Focus Sweep
    focus_sweep_min = 0        # you can tighten to, say, 10
    focus_sweep_max = 120      # or 100 if it’s always in there
    focus_sweep_step = 10      # smaller step = finer search, slower sweep

    # Exposure sweep range (depends on your lighting)
    exp_sweep_min = 0
    exp_sweep_max = 10000
    exp_sweep_step = 250

    # White balance sweep range (Kelvin)
    wb_sweep_min = 3000
    wb_sweep_max = 6000
    wb_sweep_step = 200

    # Set True when you want to tune with preview at startup
    interactive_tune: bool = True
    cam_index: int = 0          # primary (robot POV)
    cam2_index: int = 2         # secondary camera index
    frame_w: int = 1280 #3840 1280
    frame_h: int = 720 #3040 720
    use_mjpg: bool = True
    req_fps: int = 60

    # JPEG encoding
    turbojpeg_quality: int = 60
    opencv_quality: int = 60
    stream_downscale_width: int = 0

    # Detection
    target_rgb: Tuple[int, int, int] = (195, 52, 110)
    rel_tol: float = 0.20
    area_min: int = 600

    # Beam ROIs in pixel space: [min_x, max_x, min_y, max_y]
    beam_roi_global: Optional[List[int]] = None
    beam_roi_left: Optional[List[int]] = None
    beam_roi_right: Optional[List[int]] = None

    # Servo / PCA9685
    pca_channels: int = 16
    pca_address: int = 0x40
    pwm_freq: int = 50

    ch_sg90: int = 0
    ch_ms24: int = 1

    sg90_range: int = 180
    sg90_pw: Tuple[int, int] = (500, 2500)

    ms24_range: int = 200
    ms24_pw: Tuple[int, int] = (700, 2500)

    sg90_idle_deg: float = 180.0
    sg90_fire_deg: float = 0.0
    ms24_fire_deg: float = 20.0

    lens_delay: float = 1.0
    spray_time: float = 0.5
    smooth_steps: int = 12
    smooth_delay: float = 0.02
    rearm_after: float = 2.0

    # Ultrasonic
    ultra_trig: int = 20
    ultra_echo: int = 21
    ultra_hz: float = 10.0

    @classmethod
    def load(cls) -> "Config":
        """Load config.json and override defaults."""
        cfg = cls()  # defaults

        if not os.path.exists(CONFIG_FILE):
            print("[Config] No config.json found. Using defaults.")
            return cfg

        try:
            with open(CONFIG_FILE, "r") as f:
                data: Dict[str, Any] = json.load(f)
        except Exception as e:
            print(f"[Config] Error reading config.json: {e}")
            return cfg

        # Override any matching fields dynamically
        for key, val in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)

        # Convert lists → tuples where needed
        if isinstance(cfg.target_rgb, list):
            cfg.target_rgb = tuple(cfg.target_rgb)
        if isinstance(cfg.sg90_pw, list):
            cfg.sg90_pw = tuple(cfg.sg90_pw)
        if isinstance(cfg.ms24_pw, list):
            cfg.ms24_pw = tuple(cfg.ms24_pw)

        return cfg
