#!/usr/bin/env bash
# Loads a system-level "OpenASL Camera" virtual webcam device on Linux using v4l2loopback (no OBS needed).
set -euo pipefail

if ! modinfo v4l2loopback >/dev/null 2>&1; then
  echo "[OpenASL] Installing v4l2loopback-dkms..."
  sudo apt-get update && sudo apt-get install -y v4l2loopback-dkms
fi

echo "[OpenASL] Loading OpenASL Camera at /dev/video10..."
sudo modprobe -r v4l2loopback 2>/dev/null || true
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="OpenASL Camera" exclusive_caps=1
echo "[OpenASL] Ready! Select 'OpenASL Camera' in Zoom, Teams, or your browser."
