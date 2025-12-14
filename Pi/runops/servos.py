# servos.py
import sys
import time
from typing import Optional

from config import Config


class Servos:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = False
        self.kit: Optional[object] = None

        try:
            from adafruit_servokit import ServoKit
            self.kit = ServoKit(channels=cfg.pca_channels,
                                address=cfg.pca_address)
            pca = getattr(self.kit, "_pca", None)
            if pca:
                pca.frequency = cfg.pwm_freq

            # Configure shield servo (SG90)
            self.kit.servo[cfg.ch_sg90].actuation_range = cfg.sg90_range
            self.kit.servo[cfg.ch_sg90].set_pulse_width_range(*cfg.sg90_pw)

            # Configure spray actuator servo (MS24)
            self.kit.servo[cfg.ch_ms24].actuation_range = cfg.ms24_range
            self.kit.servo[cfg.ch_ms24].set_pulse_width_range(*cfg.ms24_pw)

            # Initial positions: shield at idle (open for camera), sprayer retracted
            self.kit.servo[cfg.ch_sg90].angle = cfg.sg90_idle_deg
            self.kit.servo[cfg.ch_ms24].angle = 0

            self.enabled = True
        except Exception as e:
            print(f"[Servo] Disabled: {e}", file=sys.stderr)

    def _set(self, ch: int, deg: float):
        if not self.enabled or self.kit is None:
            return
        try:
            self.kit.servo[ch].angle = float(deg)
        except Exception as e:
            return

    def smooth_move(self, ch: int, target: float,
                    steps: Optional[int] = None,
                    delay: Optional[float] = None):
        if not self.enabled or self.kit is None:
            return

        if steps is None:
            steps = self.cfg.smooth_steps
        if delay is None:
            delay = self.cfg.smooth_delay

        try:
            cur = self.kit.servo[ch].angle
        except Exception:
            cur = target

        cur = float(cur or target)
        target = float(target)

        for i in range(1, steps + 1):
            a = cur + (target - cur) * (i / steps)
            self._set(ch, a)
            time.sleep(delay)

    def _depower_channel(self, ch: int) -> None:
        """
        Cut PWM to a given channel to let the servo go limp (no holding torque).
        Used for the sg90 shield to prevent jitter / overheating.
        """
        if not self.enabled:
            return
        try:
            # Access the raw PCA9685 channel and zero its duty cycle.
            pca = getattr(self.kit, "_pca", None)
            if pca is not None:
                pca.channels[ch].duty_cycle = 0
        except Exception as e:
            print(f"[Servos] Failed to depower channel {ch}: {e}")

    # High-level shield helpers

    def open_shield(self):
        """
        Open the camera shield (for scanning / viewing) and then depower
        the sg90 so it does not jitter or cook itself.

        The next time we move it (open/close), the PWM will be re-enabled
        automatically by self._set / self.smooth_move.
        """
        if not self.enabled:
            return

        # Move to the open position
        self.smooth_move(self.cfg.ch_sg90, self.cfg.sg90_idle_deg)

        # Let the servo coast briefly (optional small wait)
        time.sleep(0.1)

        # Cut PWM on the shield channel to avoid jitter / heat
        self._depower_channel(self.cfg.ch_sg90)


    def close_shield(self):
        """
        Close the camera shield so it stays out of the way during motion/spray.
        """
        if not self.enabled:
            return
        self.smooth_move(self.cfg.ch_sg90, self.cfg.sg90_fire_deg)

    def run_sequence(self):
        """
        Spray actuation sequence.

        Assumptions:
        - Called when the robot is already at the correct target pose.
        - We do NOT reopen the shield at the end; it remains closed.
        - All angles, timings, and pulse settings come from Config.
        """
        if not self.enabled:
            return

        # Ensure shield is closed before spraying
        self.close_shield()
        time.sleep(self.cfg.lens_delay)

        # Extend sprayer
        self.smooth_move(self.cfg.ch_ms24, self.cfg.ms24_fire_deg)
        time.sleep(self.cfg.spray_time)

        # Retract sprayer
        self.smooth_move(self.cfg.ch_ms24, 0)
        time.sleep(self.cfg.lens_delay)

        # Do NOT reopen the shield here; orchestrator keeps it closed
        # after scanning and throughout motion/spray.

    def park(self):
        if not self.enabled:
            return

        # Simple safe park position
        self.smooth_move(self.cfg.ch_sg90, 0, steps=6, delay=0.03)
        self.smooth_move(self.cfg.ch_ms24, 0, steps=6, delay=0.03)
