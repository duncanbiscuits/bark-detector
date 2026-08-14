"""
Bark Detector v2 - ML edition with live graph + self-update
--------------------------------------------------------------
Listens to the microphone, runs YAMNet (a small on-device sound
classification model from Google) to check specifically for a dog
bark/dog/howl/growl sound (not just "loud"), and plays an alert when
its confidence crosses a threshold. Shows a live scrolling graph of
that confidence against the threshold line, and flashes when it fires.

On startup it also checks GitHub for a newer release, and if one is
available, downloads it, verifies its checksum, and restarts itself
on the new version. This only runs when packaged as an .exe - running
the .py directly never tries to self-update.

Security / privacy notes (kept simple and auditable):
  - 100% local and offline for detection. No telemetry, ever.
  - The only network call this app makes is a read-only check against
    GitHub's public release API (and, only if a newer version exists,
    downloading that release's .exe + checksums.txt from GitHub).
  - A downloaded update is only installed if its SHA-256 checksum
    matches the checksums.txt published alongside it in the same
    GitHub release. If it doesn't match, the update is discarded and
    the app carries on running its current version.
  - Raw audio for bark detection is only ever held in a small rolling
    memory buffer (under 2 seconds) - never written to disk or logged.
  - All local settings live in config.ini next to this script/exe.
"""

import configparser
import csv
import hashlib
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd
from ai_edge_litert.interpreter import Interpreter

if sys.platform == "win32":
    import winsound

CONFIG_FILENAME = "config.ini"
MODEL_FILENAME = "yamnet.tflite"
LABELS_FILENAME = "yamnet_class_map.csv"
VERSION_FILENAME = "version.txt"
TARGET_LABELS = ["Dog", "Bark", "Bow-wow", "Howl", "Growling"]
SAMPLE_RATE = 16000
RING_SECONDS = 1.5
INFER_WINDOW_SECONDS = 1.0
UPDATE_INTERVAL_MS = 200

GITHUB_OWNER = "duncanbiscuits"
GITHUB_REPO = "bark-detector"
ASSET_NAME = "BarkDetectorML.exe"
CHECKSUM_ASSET_NAME = "checksums.txt"
UPDATE_CHECK_TIMEOUT = 6
UPDATE_DOWNLOAD_TIMEOUT = 60

DEFAULTS = {
    "bark_threshold": "0.3",   # 0.0-1.0 confidence needed to count as a bark
    "cooldown_seconds": "5",   # minimum gap between two alerts
    "sound_file": "",          # path to a .wav to play; blank = built-in beep
    "device": "",              # blank = system default microphone
    "history_seconds": "15",   # how much time the graph shows
    "auto_update": "true",     # check GitHub for a newer release on startup
}


def base_dir():
    return os.path.dirname(os.path.abspath(sys.argv[0]))


def resource_path(name):
    """Location of bundled read-only files (model, labels, version) - works
    both as a plain script and as a PyInstaller --onefile exe."""
    base = getattr(sys, "_MEIPASS", base_dir())
    return os.path.join(base, name)


def app_version():
    try:
        with open(resource_path(VERSION_FILENAME), "r", encoding="utf-8") as f:
            return f.read().strip() or "0.0.0-dev"
    except Exception:
        return "0.0.0-dev"


def _version_tuple(v):
    v = v.strip().lstrip("vV")
    parts = []
    for p in v.split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def check_for_update(current_version):
    """Returns an update-info dict if a newer GitHub release exists, else None.
    Never raises - any network/parse problem just means "no update found"."""
    try:
        url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
        req = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github+json", "User-Agent": "BarkDetectorML"}
        )
        with urllib.request.urlopen(req, timeout=UPDATE_CHECK_TIMEOUT, context=_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        tag = data.get("tag_name", "")
        if not tag or _version_tuple(tag) <= _version_tuple(current_version):
            return None

        assets = {a["name"]: a["browser_download_url"] for a in data.get("assets", [])}
        if ASSET_NAME not in assets:
            return None

        return {
            "version": tag,
            "exe_url": assets[ASSET_NAME],
            "checksum_url": assets.get(CHECKSUM_ASSET_NAME),
        }
    except Exception as e:
        print(f"Update check skipped ({e})")
        return None


def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_update(info):
    """Downloads the new exe, verifies its checksum, and hands off to a tiny
    batch helper that swaps the file and relaunches once this process exits.
    Returns True if an update was staged (caller should exit immediately)."""
    if not getattr(sys, "frozen", False):
        print("Running from source - skipping self-update.")
        return False

    exe_path = os.path.abspath(sys.executable)
    new_path = exe_path + ".new"
    ctx = _ssl_context()

    try:
        req = urllib.request.Request(info["exe_url"], headers={"User-Agent": "BarkDetectorML"})
        with urllib.request.urlopen(req, timeout=UPDATE_DOWNLOAD_TIMEOUT, context=ctx) as resp:
            with open(new_path, "wb") as out:
                out.write(resp.read())
    except Exception as e:
        print(f"Update download failed, staying on current version: {e}")
        return False

    if info.get("checksum_url"):
        try:
            req2 = urllib.request.Request(info["checksum_url"], headers={"User-Agent": "BarkDetectorML"})
            with urllib.request.urlopen(req2, timeout=UPDATE_CHECK_TIMEOUT, context=ctx) as resp:
                checksums_text = resp.read().decode("utf-8", errors="ignore")
            expected = None
            for line in checksums_text.splitlines():
                if ASSET_NAME in line:
                    expected = line.split()[0].strip().lower()
                    break
            if expected and _sha256_of(new_path).lower() != expected:
                os.remove(new_path)
                print("Update checksum did not match - discarding update for safety.")
                return False
            if not expected:
                print("No checksum found for this release - discarding update for safety.")
                os.remove(new_path)
                return False
        except Exception as e:
            print(f"Could not verify update checksum, discarding update for safety: {e}")
            try:
                os.remove(new_path)
            except OSError:
                pass
            return False
    else:
        print("Release has no checksums.txt - discarding update for safety.")
        os.remove(new_path)
        return False

    bat_path = exe_path + ".update.bat"
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(
            "@echo off\r\n"
            "timeout /t 2 /nobreak >nul\r\n"
            f'move /Y "{new_path}" "{exe_path}"\r\n'
            f'start "" "{exe_path}"\r\n'
            'del "%~f0"\r\n'
        )
    subprocess.Popen(
        ["cmd", "/c", bat_path],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )
    print(f"Update to {info['version']} staged - restarting...")
    return True


def load_config():
    path = os.path.join(base_dir(), CONFIG_FILENAME)
    cfg = configparser.ConfigParser()
    cfg["bark"] = DEFAULTS.copy()
    if os.path.exists(path):
        cfg.read(path)
    else:
        with open(path, "w") as f:
            cfg.write(f)
    return cfg["bark"]


def load_target_indices():
    path = resource_path(LABELS_FILENAME)
    idx_by_name = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            index, _mid, display_name = row[0], row[1], row[2]
            idx_by_name[display_name] = int(index)
    return [idx_by_name[n] for n in TARGET_LABELS if n in idx_by_name]


def play_alert(sound_file):
    try:
        if sound_file and os.path.exists(sound_file):
            winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            winsound.Beep(1000, 250)
            winsound.Beep(1400, 250)
    except Exception as e:
        print(f"  ! could not play sound: {e}")


class AudioRing:
    """Small thread-safe rolling buffer of the most recent audio samples."""

    def __init__(self, seconds, rate):
        self.rate = rate
        self.buf = np.zeros(int(seconds * rate), dtype=np.float32)
        self.lock = threading.Lock()

    def write(self, chunk: np.ndarray):
        chunk = chunk.flatten().astype(np.float32)
        n = len(chunk)
        with self.lock:
            if n >= len(self.buf):
                self.buf = chunk[-len(self.buf):]
            else:
                self.buf = np.concatenate((self.buf[n:], chunk))

    def snapshot(self, seconds):
        n = int(seconds * self.rate)
        with self.lock:
            return self.buf[-n:].copy()


class BarkApp:
    def __init__(self, root, cfg, version):
        self.root = root
        self.version = version
        self.threshold = float(cfg["bark_threshold"])
        self.cooldown = float(cfg["cooldown_seconds"])
        self.sound_file = cfg["sound_file"].strip()
        self.device = cfg["device"].strip() or None
        self.history_seconds = float(cfg["history_seconds"])

        self.target_indices = load_target_indices()
        self.interpreter = Interpreter(model_path=resource_path(MODEL_FILENAME))

        self.ring = AudioRing(RING_SECONDS, SAMPLE_RATE)
        max_points = max(10, int(self.history_seconds * 1000 / UPDATE_INTERVAL_MS))
        self.history = deque(maxlen=max_points)
        self.last_alert = 0.0
        self.last_alert_str = "none yet"
        self.flash_until = 0.0

        self._build_ui()
        self._start_audio()
        self.root.after(UPDATE_INTERVAL_MS, self._tick)

    # ---------- UI ----------
    def _build_ui(self):
        self.root.title(f"Bark Detector v{self.version}")
        self.root.geometry("760x420")
        self.root.configure(bg="#1e1e1e")

        self.status_label = tk.Label(
            self.root, text="Starting...", font=("Segoe UI", 12),
            bg="#1e1e1e", fg="#eeeeee", justify="left", anchor="w",
        )
        self.status_label.pack(fill="x", padx=12, pady=(10, 4))

        self.canvas = tk.Canvas(self.root, width=730, height=300, bg="#111111",
                                 highlightthickness=0)
        self.canvas.pack(padx=12, pady=6)

        self.detail_label = tk.Label(
            self.root, text="", font=("Segoe UI", 10),
            bg="#1e1e1e", fg="#9a9a9a", justify="left", anchor="w",
        )
        self.detail_label.pack(fill="x", padx=12, pady=(0, 10))

    def _start_audio(self):
        def callback(indata, frames, time_info, status):
            if status:
                pass  # ignore minor buffer warnings
            self.ring.write(indata[:, 0])

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=int(SAMPLE_RATE * 0.1),
            device=self.device,
            callback=callback,
        )
        self.stream.start()

    # ---------- main loop ----------
    def _tick(self):
        snap = self.ring.snapshot(INFER_WINDOW_SECONDS)
        score = 0.0
        if len(snap) >= SAMPLE_RATE // 2:
            self.interpreter.resize_tensor_input(0, [len(snap)])
            self.interpreter.allocate_tensors()
            self.interpreter.set_tensor(0, snap)
            self.interpreter.invoke()
            scores = self.interpreter.get_tensor(203)
            score = float(scores[:, self.target_indices].max())

        now = time.monotonic()
        self.history.append(score)

        triggered = False
        if score >= self.threshold and (now - self.last_alert) >= self.cooldown:
            self.last_alert = now
            self.last_alert_str = datetime.now().strftime("%H:%M:%S")
            self.flash_until = now + 0.6
            triggered = True
            play_alert(self.sound_file)

        self._redraw(score, triggered)
        self.root.after(UPDATE_INTERVAL_MS, self._tick)

    # ---------- drawing ----------
    def _redraw(self, score, triggered):
        c = self.canvas
        c.delete("all")
        w = int(c["width"])
        h = int(c["height"])
        pad = 20

        flashing = time.monotonic() < self.flash_until
        bg = "#3a1010" if flashing else "#111111"
        c.configure(bg=bg)

        # threshold line
        thresh_y = pad + (1 - self.threshold) * (h - 2 * pad)
        c.create_line(pad, thresh_y, w - pad, thresh_y, fill="#e05050", dash=(4, 3))
        c.create_text(w - pad, thresh_y - 10, text=f"threshold {self.threshold:.2f}",
                       fill="#e05050", anchor="e", font=("Segoe UI", 9))

        # confidence line
        pts = list(self.history)
        if len(pts) >= 2:
            n = len(pts)
            step = (w - 2 * pad) / max(1, (self.history.maxlen - 1))
            x0 = w - pad - (n - 1) * step
            coords = []
            for i, v in enumerate(pts):
                x = x0 + i * step
                y = pad + (1 - min(max(v, 0), 1)) * (h - 2 * pad)
                coords.extend([x, y])
            c.create_line(*coords, fill="#4ea1ff", width=2, smooth=True)

        color = "#ff5555" if flashing else "#4ea1ff"
        c.create_text(pad, pad - 5, text=f"bark confidence: {score:.2f}",
                       fill=color, anchor="nw", font=("Segoe UI", 11, "bold"))

        state = "BARK DETECTED - alert played" if flashing else "listening..."
        self.status_label.config(
            text=f"{state}    |    last alert: {self.last_alert_str}",
            fg="#ff8080" if flashing else "#eeeeee",
        )
        self.detail_label.config(
            text=f"v{self.version}  threshold={self.threshold:.2f}  cooldown={self.cooldown:.0f}s  "
                 f"mic={self.device or 'system default'}  (edit config.ini to change)"
        )


def main():
    version = app_version()
    cfg = load_config()

    if cfg.get("auto_update", "true").strip().lower() in ("1", "true", "yes", "on"):
        update_info = check_for_update(version)
        if update_info and apply_update(update_info):
            return  # a new version is about to take over; exit quietly

    root = tk.Tk()
    app = BarkApp(root, cfg, version)

    def on_close():
        try:
            app.stream.stop()
            app.stream.close()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
