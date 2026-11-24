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
import os
from pathlib import Path
import platform
import struct
import sys
from typing import Optional

import cv2
import numpy as np
import pyvirtualcam
import websockets

WIDTH, HEIGHT, FPS = 1280, 720, 30
latest_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
connected_clients = 0


def ensure_windows_16x9_unitycapture() -> None:
    """
    Ensures UnityCaptureFilter64.dll advertises 16:9 resolutions (1280x720 primary)
    across all DirectShow stream capability slots so downstream apps (WebRTC, Discord,
    Windows Camera, Zoom) never negotiate a 4:3 (640x480 / 960x720) pin mode.
    Registers the 16:9 DLL path in HKCU\\Software\\Classes\\CLSID (no Admin required).
    """
    if platform.system().lower() != "windows":
        return

    try:
        orig_table = [
            (1920, 1080), (1280, 720), (960, 540), (640, 360),
            (480, 270), (256, 144), (2560, 1440), (3840, 2160),
            (1440, 1080), (960, 720), (640, 480), (480, 360),
            (320, 240), (192, 144), (1920, 1440), (2880, 2160),
            (1920, 1200), (1280, 800), (2880, 1800), (2560, 1600),
            (1680, 1050), (1440, 900), (0, 0),
        ]
        new_16x9_table = [
            (1280, 720), (1920, 1080), (960, 540), (640, 360),
            (1280, 720), (1280, 720), (2560, 1440), (3840, 2160),
            (1280, 720), (1280, 720), (1280, 720), (640, 360),
            (640, 360), (640, 360), (1280, 720), (1920, 1080),
            (1280, 720), (1280, 720), (1920, 1080), (1280, 720),
            (1280, 720), (1280, 720), (0, 0),
        ]
        fmt = "<" + "ii" * len(orig_table)
        orig_bytes = struct.pack(fmt, *[v for pair in orig_table for v in pair])
        new_bytes = struct.pack(fmt, *[v for pair in new_16x9_table for v in pair])

        prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        sys_dll = Path(prog_files) / "OpenASL" / "driver" / "UnityCaptureFilter64.dll"
        local_driver_dir = Path(__file__).resolve().parent / ".driver"
        local_dll = local_driver_dir / "UnityCaptureFilter64.dll"

        raw_data: Optional[bytes] = None
        if local_dll.exists():
            raw_data = local_dll.read_bytes()
        elif sys_dll.exists():
            raw_data = sys_dll.read_bytes()

        if not raw_data:
            return

        if orig_bytes in raw_data:
            patched = raw_data.replace(orig_bytes, new_bytes, 1)
            local_driver_dir.mkdir(parents=True, exist_ok=True)
            local_dll.write_bytes(patched)
            try:
                sys_dll.write_bytes(patched)
            except Exception:
                pass
        elif new_bytes in raw_data and not local_dll.exists():
            local_driver_dir.mkdir(parents=True, exist_ok=True)
            local_dll.write_bytes(raw_data)

        target_dll = local_dll if local_dll.exists() else sys_dll
        if target_dll.exists():
            import winreg  # type: ignore

            clsid_key = r"Software\Classes\CLSID\{5C2CD55C-92AD-4999-8666-912BD3E70010}\InprocServer32"
            with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, clsid_key, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "", 0, winreg.REG_SZ, str(target_dll))
                winreg.SetValueEx(key, "ThreadingModel", 0, winreg.REG_SZ, "Both")
    except Exception as exc:
        print(f"[OpenASL Bridge] Note: 16:9 driver pin setup skipped ({exc})")


def make_standby_frame() -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (13, 10, 8), dtype=np.uint8)
    cv2.putText(
        frame,
        "OPENASL VIRTUAL CAMERA READY",
        (335, 345),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (239, 236, 234),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Enable Webcam PiP + Virtual Cam in OpenASL to stream live",
        (285, 395),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (174, 158, 148),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def open_virtual_camera(preferred_backend: Optional[str], device: Optional[str]) -> pyvirtualcam.Camera:
    system = platform.system().lower()
    if system == "windows":
        ensure_windows_16x9_unitycapture()
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


async def frame_receiver(websocket, frame_event: asyncio.Event) -> None:
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
                    frame_event.set()
    except websockets.ConnectionClosed:
        pass
    finally:
        connected_clients = max(0, connected_clients - 1)
        print("[OpenASL Bridge] Client disconnected.")
        if connected_clients == 0:
            latest_frame = make_standby_frame()
            frame_event.set()


async def camera_pump(cam: pyvirtualcam.Camera, frame_event: asyncio.Event) -> None:
    interval = 1.0 / FPS
    while True:
        try:
            await asyncio.wait_for(frame_event.wait(), timeout=interval)
            frame_event.clear()
        except asyncio.TimeoutError:
            pass
        cam.send(latest_frame)


async def run_bridge(host: str, port: int, backend: Optional[str], device: Optional[str]) -> None:
    global latest_frame
    latest_frame = make_standby_frame()
    frame_event = asyncio.Event()

    with open_virtual_camera(backend, device) as cam:
        print(f"[OpenASL VirtualCam] Active -> {cam.device} (backend={cam.backend}, {WIDTH}x{HEIGHT}@{FPS}fps)")
        print(f"[OpenASL Bridge] Listening on ws://{host}:{port} (compression=None, event-driven)")
        async with websockets.serve(
            lambda ws: frame_receiver(ws, frame_event),
            host,
            port,
            max_size=4 * 1024 * 1024,
            compression=None,
        ):
            await camera_pump(cam, frame_event)


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
