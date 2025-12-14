#!/usr/bin/env bash
set -e

PORT=8000

# Always operate from the script's directory (so http.server serves this folder)
cd "$(dirname "$0")"

echo "Killing anything on port $PORT..."
sudo fuser -k ${PORT}/tcp 2>/dev/null || true

echo "Starting python3 -m http.server ${PORT} ..."
python3 -m http.server ${PORT} &
HTTP_PID=$!

# Give the server a moment to start
sleep 1

echo "Checking local HTTP server with curl..."
curl http://127.0.0.1:${PORT}/ | head

echo "Resetting Tailscale Serve/Funnel config..."
sudo tailscale serve reset || true
sudo tailscale funnel reset 2>/dev/null || true

echo "Enabling Tailscale Funnel on port ${PORT}..."
sudo tailscale funnel ${PORT}

echo
echo "Done. HTTP server PID: ${HTTP_PID}"
echo "You should now be able to access the funnel URL shown by `tailscale funnel ${PORT}` or `tailscale funnel status`."
