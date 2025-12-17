import json
import time
import threading
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Dict, Any, Optional
from socket import error as SocketError

from config import Config


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class BackendHandler(BaseHTTPRequestHandler):
    """
    HTTP backend:

      GET /           -> HTML UI
      GET /stream     -> MJPEG from primary camera (robot view)
      GET /stream2    -> MJPEG from secondary camera (overview)
      GET /status     -> JSON status from orchestrator
    """

    frame_provider: Optional[Callable[[], bytes]] = None
    frame_provider2: Optional[Callable[[], bytes]] = None
    status_provider: Optional[Callable[[], Dict[str, Any]]] = None

    server_version = "BridgeBotHTTP/0.3"

    def log_message(self, format: str, *args) -> None:
        # Quiet logs
        return

    def _write_headers(self, code: int, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]

        if path in ("", "/"):
            self._serve_index()
        elif path == "/stream":
            self._serve_mjpeg(primary=True)
        elif path == "/stream2":
            self._serve_mjpeg(primary=False)
        elif path == "/status":
            self._serve_status()
        else:
            self.send_error(404, "Not Found")

    def _serve_index(self) -> None:
        self._write_headers(200, "text/html; charset=utf-8")
        # Layout logic:
        #  - #secondary-pane is always visible (cam2)
        #  - #primary-pane (cam0) is hidden unless scan_active == true
        html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Bridge Robot – Rust Cameras</title>
  <style>
    :root {
      color-scheme: dark;
    }
    body {
      margin: 0;
      padding: 0;
      background: #0f1115;
      color: #e5e7eb;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .wrapper {
      box-sizing: border-box;
      padding: 10px;
      height: calc(100vh - 32px);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .streams.single {
      flex: 1 1 0;
      display: flex;
      flex-direction: column;
    }
    .streams.dual {
      flex: 1 1 0;
      display: flex;
      flex-direction: row;
      gap: 10px;
    }
    .pane {
      flex: 1 1 0;
      display: flex;
      flex-direction: column;
      background: #111827;
      border-radius: 10px;
      border: 1px solid #1f2937;
      overflow: hidden;
      min-height: 0;
    }
    .pane-header {
      padding: 6px 10px;
      font-size: 14px;
      background: #111827;
      border-bottom: 1px solid #1f2937;
    }
    .pane-body {
      flex: 1 1 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #000;
    }
    .pane-body img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      background: #000;
    }
    .hidden {
      display: none !important;
    }
    .status-bar {
      height: 22px;
      font-size: 12px;
      display: flex;
      flex-direction: row;
      align-items: center;
      gap: 16px;
      padding: 4px 8px;
      border-radius: 6px;
      background: #111827;
      border: 1px solid #1f2937;
    }
    .status-item span.label {
      color: #9ca3af;
      margin-right: 2px;
    }
    .badge {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      font-size: 11px;
      border: 1px solid #374151;
      background: #111827;
    }
    .badge.scanning {
      border-color: #f97316;
      background: #451a03;
      color: #fed7aa;
    }
  </style>
</head>
<body>
  <div class="wrapper">
    <div id="streams" class="streams single">
      <div id="secondary-pane" class="pane">
        <div class="pane-header">Secondary / Overview Camera</div>
        <div class="pane-body">
          <img src="/stream2" alt="Secondary camera stream">
        </div>
      </div>
      <div id="primary-pane" class="pane hidden">
        <div class="pane-header">Primary Robot View (Active During Scan)</div>
        <div class="pane-body">
          <img src="/stream" alt="Primary camera stream">
        </div>
      </div>
    </div>
    <div class="status-bar" id="status-bar">
      <div class="status-item">
        <span class="label">Mode:</span>
        <span id="status-mode" class="badge">idle</span>
      </div>
      <div class="status-item">
        <span class="label">Label:</span>
        <span id="status-label">INIT</span>
      </div>
      <div class="status-item">
        <span class="label">FPS:</span>
        <span id="status-fps">0.0</span>
      </div>
      <div class="status-item">
        <span class="label">Boxes:</span>
        <span id="status-boxes">0</span>
      </div>
      <div class="status-item">
        <span class="label">Dist (in):</span>
        <span id="status-dist">-</span>
      </div>
    </div>
  </div>
  <script>
    function applyLayoutFromStatus(data) {
      const streams = document.getElementById('streams');
      const primaryPane = document.getElementById('primary-pane');
      const modeBadge = document.getElementById('status-mode');

      const scanning = !!data.scan_active;

      if (scanning) {
        streams.classList.remove('single');
        streams.classList.add('dual');
        primaryPane.classList.remove('hidden');
        modeBadge.textContent = data.scan_state || 'scanning';
        modeBadge.classList.add('scanning');
      } else {
        streams.classList.remove('dual');
        streams.classList.add('single');
        primaryPane.classList.add('hidden');
        modeBadge.textContent = data.scan_state || 'idle';
        modeBadge.classList.remove('scanning');
      }
    }

    async function pollStatus() {
      try {
        const res = await fetch('/status', { cache: 'no-store' });
        if (!res.ok) throw new Error('bad status');
        const data = await res.json();

        document.getElementById('status-label').textContent = data.label ?? 'n/a';
        document.getElementById('status-fps').textContent = (data.fps ?? 0).toFixed(1);
        document.getElementById('status-boxes').textContent = data.last_boxes ?? 0;
        const d = data.distance_from_bridge_in;
        document.getElementById('status-dist').textContent =
          (d === null || d === undefined) ? '-' : d.toFixed(2);

        applyLayoutFromStatus(data);
      } catch (e) {
        // ignore errors; retry
      } finally {
        setTimeout(pollStatus, 1000);
      }
    }
    pollStatus();
  </script>
</body>
</html>
"""
        self.wfile.write(html.encode("utf-8"))

    def _serve_mjpeg(self, primary: bool) -> None:
        provider = self.frame_provider if primary else self.frame_provider2
        if provider is None:
            self.send_error(503, "Stream provider not available")
            return

        boundary = "frame"
        self._write_headers(200, f"multipart/x-mixed-replace; boundary={boundary}")

        try:
            while True:
                frame = b""
                try:
                    frame = provider() or b""
                except Exception:
                    frame = b""

                if not frame:
                    time.sleep(0.05)
                    continue

                try:
                    self.wfile.write(b"--" + boundary.encode("ascii") + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(b"Content-Length: " + str(len(frame)).encode("ascii") + b"\r\n")
                    self.wfile.write(b"\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break

                time.sleep(0.03)
        except Exception:
            pass

    def _serve_status(self) -> None:
        provider = self.status_provider
        if provider is None:
            data = {"error": "no_status_provider"}
        else:
            try:
                data = provider()
            except Exception as e:
                data = {"error": str(e)}

        payload = json.dumps(data).encode("utf-8")
        self._write_headers(200, "application/json; charset=utf-8")
        try:
            self.wfile.write(payload)
        except BrokenPipeError:
            # Client disconnected mid-write; ignore.
            return



class Website:
    """
    Thin wrapper to run the HTTP backend in a background thread.
    """

    def __init__(
        self,
        cfg: Config,
        frame_provider: Callable[[], bytes],
        status_provider: Callable[[], Dict[str, Any]],
        second_frame_provider: Callable[[], bytes],
    ):
        self.cfg = cfg
        self._server = ThreadingHTTPServer((cfg.host, cfg.port), BackendHandler)
        self._thread: threading.Thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )

        BackendHandler.frame_provider = frame_provider
        BackendHandler.frame_provider2 = second_frame_provider
        BackendHandler.status_provider = status_provider

    def start(self) -> None:
        self._thread.start()
        print(
            f"[Backend] http://{self.cfg.host}:{self.cfg.port} "
            f"endpoints: /stream, /stream2, /status\n"
            "Live Stream Website: https://rit-bridgerobot-live.duckdns.org"
        )

    def stop(self) -> None:
        try:
            self._server.shutdown()
        except Exception:
            pass
        try:
            self._server.server_close()
        except Exception:
            pass
