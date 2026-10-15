"""Receive Quest RGB frames through scrcpy's H.264 camera stream over ADB.

Uses the pinned scrcpy 5.0.1 server, FFmpeg, and NumPy. No Android APK or
virtual webcam driver is needed. Frame times are PC decode times, NOT sensor
timestamps. Only the newest decoded frame is retained for the consumer.
"""

import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TOOLS = Path.home() / ".local/share/quest-camera-tools"
SCRCPY_VERSION = "5.0.1"


@dataclass(frozen=True)
class CameraFrame:
    bgr: np.ndarray
    number: int
    received_monotonic_ns: int


class QuestCamera:
    def __init__(self, eye="left", width=1280, height=960, fps=30,
                 bitrate=8_000_000, serial=None, adb=None, server=None, ffmpeg=None):
        self.width, self.height = width, height
        self.eye, self.fps, self.bitrate = eye, fps, bitrate
        self.adb = str(adb or shutil.which("adb") or TOOLS / "platform-tools/adb")
        self.server = Path(server or TOOLS / f"scrcpy-linux-x86_64-v{SCRCPY_VERSION}/scrcpy-server")
        self.ffmpeg = str(ffmpeg or shutil.which("ffmpeg") or "ffmpeg")
        self.serial = serial
        self.port = None
        self.remote = None
        self.camera_id = None
        self.capture = self.decoder = self.worker = None
        self.log = tempfile.TemporaryFile(mode="w+b")
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.latest = None
        self.error = None

    def _adb(self, *args, timeout=10):
        selector = ["-s", self.serial] if self.serial else []
        return subprocess.run([self.adb, *selector, *args], check=True,
                              capture_output=True, text=True, timeout=timeout).stdout.strip()

    def start(self):
        try:
            if not self.server.is_file():
                raise RuntimeError(f"Missing scrcpy {SCRCPY_VERSION} server: {self.server}")
            if not self.serial:
                devices = [line.split()[0] for line in self._adb("devices").splitlines()[1:]
                           if len(line.split()) >= 2 and line.split()[1] == "device"]
                if len(devices) != 1:
                    raise RuntimeError("Connect and authorize exactly one Quest, or pass --serial.\n"
                                       + self._adb("devices", "-l"))
                self.serial = devices[0]
            # Identify the passthrough cameras by Meta's tags, not numeric IDs.
            metadata = self._adb("shell", "dumpsys", "media.camera")
            sections = re.split(r"== Camera HAL device .*?/([^/\s]+) \(v[^)]*\) static information: ==", metadata)
            wanted_position = 0 if self.eye == "left" else 1
            for index in range(1, len(sections), 2):
                camera_id, body = sections[index:index + 2]
                source = re.search(r"com\.meta\.extra_metadata\.camera_source[^\n]*\n\s*\[(\d+)", body)
                position = re.search(r"com\.meta\.extra_metadata\.position[^\n]*\n\s*\[(\d+)", body)
                if source and position and int(source[1]) == 0 and int(position[1]) == wanted_position:
                    self.camera_id = camera_id
                    break
            if self.camera_id is None:
                raise RuntimeError(f"Cannot identify the {self.eye} RGB camera from Meta vendor tags.")

            session = secrets.randbelow(0x7ffffffe) + 1
            socket_name = f"scrcpy_{session:08x}"
            self.remote = f"/data/local/tmp/wilor-scrcpy-{session:08x}.jar"
            self._adb("push", str(self.server), self.remote)
            self.port = int(self._adb("forward", "tcp:0", f"localabstract:{socket_name}"))
            options = [f"scid={session:08x}", "tunnel_forward=true", "video_source=camera",
                       f"camera_id={self.camera_id}", f"camera_size={self.width}x{self.height}",
                       f"camera_fps={self.fps}", "video_codec=h264", f"video_bit_rate={self.bitrate}",
                       "audio=false", "control=false", "raw_stream=true", "cleanup=false",
                       "video_codec_options=i-frame-interval=1", "log_level=info"]
            self.capture = subprocess.Popen(
                [self.adb, "-s", self.serial, "shell", f"CLASSPATH={self.remote}",
                 "app_process", "/", "com.genymobile.scrcpy.Server", SCRCPY_VERSION, *options],
                stdout=self.log, stderr=self.log)
            deadline = time.monotonic() + 10
            while socket_name not in self._adb("shell", "cat", "/proc/net/unix"):
                if self.capture.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("scrcpy camera server did not start. " + self.diagnostics())
                time.sleep(0.1)

            # Raw H.264 avoids recording-container buffering. FFmpeg continuously
            # drains the ADB socket even when WiLoR runs more slowly than capture.
            self.decoder = subprocess.Popen(
                [self.ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin",
                 "-probesize", "32", "-analyzeduration", "0", "-flags", "low_delay",
                 "-threads", "1", "-f", "h264", "-i", f"tcp://127.0.0.1:{self.port}",
                 "-an", "-vf", f"scale={self.width}:{self.height}",
                 "-threads", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"],
                stdout=subprocess.PIPE, stderr=self.log)
            self.worker = threading.Thread(target=self._read_frames, daemon=True)
            self.worker.start()
            return self
        except BaseException:
            self.close()
            raise

    def _read_frames(self):
        size = self.width * self.height * 3
        number = 0
        try:
            while not self.stopped.is_set():
                data = bytearray()
                while len(data) < size:
                    chunk = self.decoder.stdout.read(size - len(data))
                    if not chunk:
                        raise EOFError("Quest camera stream ended. " + self.diagnostics())
                    data.extend(chunk)
                number += 1
                frame = CameraFrame(np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3),
                                    number, time.monotonic_ns())
                with self.condition:
                    self.latest = frame
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def read(self, after=0, timeout=5):
        """Wait for a fresh frame; never replay the last frame after a failure."""
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.error or self.stopped.is_set() or
                (self.latest is not None and self.latest.number > after), timeout)
            if self.error:
                raise RuntimeError(str(self.error)) from self.error
            if self.stopped.is_set():
                raise RuntimeError("Camera has been closed.")
            if not ready:
                raise TimeoutError("No new Quest camera frame. " + self.diagnostics())
            return self.latest

    def diagnostics(self):
        if self.log.closed:
            return ""
        # pread does not change the file offset used by the subprocess writers.
        import os
        length = os.fstat(self.log.fileno()).st_size
        return os.pread(self.log.fileno(), min(length, 6000), max(0, length - 6000)).decode(errors="replace")

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        # Closing the decoder's TCP socket also makes the Android encoder exit.
        for process in (self.decoder, self.capture):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        if self.worker:
            self.worker.join(timeout=3)
        if self.decoder and self.decoder.stdout:
            self.decoder.stdout.close()
        if self.port:
            try:
                self._adb("forward", "--remove", f"tcp:{self.port}", timeout=3)
            except (subprocess.SubprocessError, OSError):
                pass
            self.port = None
        if self.remote:
            try:
                self._adb("shell", "rm", "-f", self.remote, timeout=3)
            except (subprocess.SubprocessError, OSError):
                pass
            self.remote = None
        self.log.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
