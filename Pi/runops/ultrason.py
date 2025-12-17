# ultrason.py
import sys
import time
import threading
from typing import Optional

from config import Config

try:
    import lgpio
except Exception as _e:
    lgpio = None
    print(f"[Ultrasonic] lgpio not available: {_e}", file=sys.stderr)


class UltraSon:
    def __init__(self, cfg: Config):
        if lgpio is None:
            raise RuntimeError("lgpio required for ultrasonic on Pi 5")

        self.cfg = cfg
        self.trig = int(cfg.ultra_trig)
        self.echo = int(cfg.ultra_echo)
        self.hz = float(cfg.ultra_hz)

        self._chip = lgpio.gpiochip_open(0)

        try:
            lgpio.gpio_claim_output(self._chip, self.trig)
            lgpio.gpio_claim_input(self._chip, self.echo)
            lgpio.gpio_write(self._chip, self.trig, 0)
        except Exception as e:
            print(f"[Ultrasonic] GPIO claim failed: {e}", file=sys.stderr)

        self._distance_in: Optional[float] = None
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def _read_once(self, timeout: float = 2.0) -> Optional[float]:
        #print("[Debug] Starting ultrasonic pulse")
        lgpio.gpio_write(self._chip, self.trig, 0)
        time.sleep(0.0002)
        lgpio.gpio_write(self._chip, self.trig, 1)
        time.sleep(0.00005)
        lgpio.gpio_write(self._chip, self.trig, 0)

        deadline = time.time() + timeout

        # Wait for echo to go HIGH
        while lgpio.gpio_read(self._chip, self.echo) == 0:
            if time.time() > deadline:
                #print("[Debug] Echo HIGH timeout")
                return None
        t0 = time.time()

        # Wait for echo to go LOW
        while lgpio.gpio_read(self._chip, self.echo) == 1:
            if time.time() > deadline:
                #print("[Debug] Echo LOW timeout")
                return None
        t1 = time.time()

        dt = t1 - t0
        distance = (dt * 34300.0 / 2.0) / 2.54  # cm → inches
        #print(f"[Debug] Echo duration: {dt:.6f} sec → Distance: {distance:.2f} in")
        #time.sleep(1)
        return distance

    def latest_inches(self) -> Optional[float]:
        return self._distance_in

    def _worker(self):
        print("[Debug] Ultrasonic worker thread started")
        period = 1.0 / self.hz if self.hz > 0 else 0.1
        while not self._stop.is_set():
            d = self._read_once()
            if d is not None:
                self._distance_in = round(d, 2)
                #print(f"[Ultrasonic] Updated distance: {self._distance_in:.2f} in")
            #time.sleep(period)

    def start(self):
        if self._t and self._t.is_alive():
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()
        print(f"[Ultrasonic] running (TRIG={self.trig}, ECHO={self.echo}, {self.hz} Hz)")

    def stop(self):
        self._stop.set()
        if self._t and self._t.is_alive():
            self._t.join(timeout=1.0)
        try:
            lgpio.gpio_write(self._chip, self.trig, 0)
        except Exception:
            pass
        try:
            lgpio.gpio_free(self._chip, self.trig)
            lgpio.gpio_free(self._chip, self.echo)
        except Exception:
            pass
        try:
            lgpio.gpiochip_close(self._chip)
        except Exception:
            pass
