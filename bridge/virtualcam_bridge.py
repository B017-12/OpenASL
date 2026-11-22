#!/usr/bin/env python3
"""
OpenASL Virtual Camera Bridge.
Receives 1280x720 JPEG frames from the WebAI HTML5 <canvas> over ws://127.0.0.1:8765
and writes them directly to a system-level Virtual Webcam device.

Backend Priority:
  - Windows: unitycapture (UnityCaptureFilter64.dll registered via regsvr32)
  - Linux:   v4l2loopback (/dev/video*)
  - macOS:   CoreMediaIO / pyvirtualcam default
"""

import argparse
import asyncio
import platform
import sys
from typing import Optional

import cv2
import numpy as np
import pyvirtualcam
import websockets

WIDTH, HEIGHT, FPS = 1280, 720, 30
latest_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
connected_clients = 0


def make_standby_frame() -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (17, 19, 23), dtype=np.uint8)
    cv2.putText(
        frame,
        "OpenASL Virtual Camera Ready",
        (350, 345),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (243, 241, 236),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Enable Webcam PiP + Virtual Cam in OpenASL to stream live",
        (285, 395),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (168, 164, 156),
        1,
        cv2.LINE_AA,
    )
    return frame


def open_virtual_camera(preferred_backend: Optional[str], device: Optional[str]) -> pyvirtualcam.Camera:
    system = platform.system().lower()
    if preferred_backend:
        candidates = [preferred_backend]
    elif system == "windows":
        # Prefer UnityCapture standalone DirectShow DLL, falling back to default pyvirtualcam backend
        candidates = ["unitycapture", None]
    elif system == "linux":
        candidates = ["v4l2loopback"]
    else:
        candidates = [None]

    last_err: Optional[Exception] = None
    for backend in candidates:
        try:
            kwargs = {
                "width": WIDTH,
                "height": HEIGHT,
                "fps": FPS,
                "fmt": pyvirtualcam.PixelFormat.RGB,
            }
            if backend:
                kwargs["backend"] = backend
            if device:
                kwargs["device"] = device
            return pyvirtualcam.Camera(**kwargs)
        except Exception as err:
            last_err = err

    hint = (
        "Run bridge/install_windows_vcam.ps1 (Administrator) to register UnityCaptureFilter64.dll."
        if system == "windows"
        else "Run bridge/setup_linux_vcam.sh to load the v4l2loopback virtual camera kernel module."
        if system == "linux"
        else "Install a virtual camera backend supported by pyvirtualcam."
    )
    raise RuntimeError(f"Could not open virtual camera ({last_err}).\nHint: {hint}")


async def frame_receiver(websocket) -> None:
    global latest_frame, connected_clients
    connected_clients += 1
    print(f"[OpenASL Bridge] Client connected ({websocket.remote_address})")
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                arr = np.frombuffer(message, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    if bgr.shape[1] != WIDTH or bgr.shape[0] != HEIGHT:
                        bgr = cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
                    latest_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients = max(0, connected_clients - 1)
        print("[OpenASL Bridge] Client disconnected.")
        if connected_clients == 0:
            latest_frame = make_standby_frame()


async def camera_pump(cam: pyvirtualcam.Camera) -> None:
    interval = 1.0 / FPS
    while True:
        cam.send(latest_frame)
        await asyncio.sleep(interval)


async def run_bridge(host: str, port: int, backend: Optional[str], device: Optional[str]) -> None:
    global latest_frame
    latest_frame = make_standby_frame()

    with open_virtual_camera(backend, device) as cam:
        print(f"[OpenASL VirtualCam] Active -> {cam.device} (backend={cam.backend}, {WIDTH}x{HEIGHT}@{FPS}fps)")
        print(f"[OpenASL Bridge] Listening on ws://{host}:{port}")
        async with websockets.serve(frame_receiver, host, port, max_size=4 * 1024 * 1024):
            await camera_pump(cam)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenASL Virtual Camera Bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backend", default=None, help="Force pyvirtualcam backend (e.g. unitycapture, v4l2loopback)")
    parser.add_argument("--device", default=None, help="Optional device path or name")
    args = parser.parse_args()

    try:
        asyncio.run(run_bridge(args.host, args.port, args.backend, args.device))
    except KeyboardInterrupt:
        print("\n[OpenASL Bridge] Stopped.")
    except Exception as exc:
        print(f"[OpenASL Bridge] Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
