"""
Bark Detector v2 - ML edition with live graph + self-update
--------------------------------------------------------------
Listens to the microphone, runs YAMNet (a small on-device sound
classification model from Google) to check specifically for a dog
bark/dog/howl/growl sound (not just "loud"), and plays an alert when
its confidence crosses a threshold. Shows a live scrolling graph of
that confidence against the threshold line, and flashes when it fires.

You can tick which alert sound files to use (played in order or at
random), pick the microphone, and drag the threshold/cooldown sliders
to tune everything live, all from the app window - no need to edit
config.ini by hand for any of these.

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
import random
import shutil
import ssl
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request
from collections import deque
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd
from ai_edge_litert.interpreter import Interpreter

if sys.platform == "win32":
    import winsound

CONFIG_FILENAME = "config.ini"
MODEL_FILENAME = "yamnet.tflite"
LABELS_FILENAME = "yamnet_class_map.csv"
VERSION_FILENAME = "version.txt"
SOUNDS_DIRNAME = "sounds"
DEFAULT_SOUNDS_DIRNAME = "default_sounds"
TARGET_LABELS = ["Dog", "Bark", "Bow-wow", "Howl", "Growling"]
SAMPLE_RATE = 16000
RING_SECONDS = 1.5
INFER_WINDOW_SECONDS = 1.0
UPDATE_INTERVAL_MS = 200
DEFAULT_MIC_LABEL = "System default"

GITHUB_OWNER = "duncanbiscuits"
GITHUB_REPO = "bark-detector"
ASSET_NAME = "BarkDetectorML.exe"
CHECKSUM_ASSET_NAME = "checksums.txt"
UPDATE_CHECK_TIMEOUT = 6
UPDATE_DOWNLOAD_TIMEOUT = 60

DEFAULTS = {
    "bark_threshold": "0.3",           # 0.0-1.0 confidence needed to count as a bark
    "cooldown_seconds": "5",           # minimum gap between two alerts
    "sound_file": "",                  # legacy single-sound setting, kept for compatibility
    "device": "",                      # blank = system default microphone, else a device index
    "history_seconds": "15",           # how much time the graph shows
    "auto_update": "true",             # check GitHub for a newer release on startup
    "sound_mode": "order",             # "order" = cycle through ticked sounds, "random" = pick one at random
    "sound_files_enabled": "sit-down.wav",  # comma list of ticked filenames inside the sounds/ folder
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


def list_input_devices():
    """Returns [(index, name), ...] for devices with at least one input channel."""
    devices = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                devices.append((i, d.get("name", f"Device {i}")))
    except Exception as e:
        print(f"  ! could not list audio devices: {e}")
    return devices


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
        self.history_seconds = float(cfg["history_seconds"])
        self._cfg_raw = dict(cfg)

        self.input_devices = list_input_devices()
        self.device = self._resolve_initial_device(cfg["device"].strip())

        self.sounds_dir = os.path.join(base_dir(), SOUNDS_DIRNAME)
        os.makedirs(self.sounds_dir, exist_ok=True)
        self.sound_mode = cfg.get("sound_mode", "order").strip() or "order"
        self._enabled_from_cfg = [
            s.strip() for s in cfg.get("sound_files_enabled", "").split(",") if s.strip()
        ]
        self._play_index = 0
        self.available_sounds = []
        self.sound_vars = {}

        # First run: seed the sounds folder with the bundled default sounds
        # (sit-down.wav, sit-down-sorry-dog.wav) so the app works out of the
        # box. Only happens once - if the folder already has files, or the
        # user has emptied it on purpose, we leave it alone.
        if not os.listdir(self.sounds_dir):
            default_sounds_src = resource_path(DEFAULT_SOUNDS_DIRNAME)
            if os.path.isdir(default_sounds_src):
                for name in os.listdir(default_sounds_src):
                    if name.lower().endswith(".wav"):
                        try:
                            shutil.copy(os.path.join(default_sounds_src, name),
                                        os.path.join(self.sounds_dir, name))
                        except Exception:
                            pass

        # One-time convenience: also honour the old single "sound_file"
        # setting if someone still has it set from an older version.
        legacy = cfg.get("sound_file", "").strip()
        if legacy and os.path.exists(legacy) and not os.listdir(self.sounds_dir):
            try:
                shutil.copy(legacy, os.path.join(self.sounds_dir, os.path.basename(legacy)))
            except Exception:
                pass

        self.target_indices = load_target_indices()
        self.interpreter = Interpreter(model_path=resource_path(MODEL_FILENAME))

        self.ring = AudioRing(RING_SECONDS, SAMPLE_RATE)
        max_points = max(10, int(self.history_seconds * 1000 / UPDATE_INTERVAL_MS))
        self.history = deque(maxlen=max_points)
        self.last_alert = 0.0
        self.last_alert_str = "none yet"
        self.flash_until = 0.0
        self.mic_error = None

        self._build_ui()
        self._refresh_sound_list()
        self._start_audio()
        self.root.after(UPDATE_INTERVAL_MS, self._tick)

    # ---------- device helpers ----------
    def _resolve_initial_device(self, cfg_device):
        if not cfg_device:
            return None
        try:
            idx = int(cfg_device)
            if any(i == idx for i, _ in self.input_devices):
                return idx
        except ValueError:
            pass
        return None  # saved device no longer exists - fall back to default

    def _mic_display_name(self):
        if self.device is None:
            return DEFAULT_MIC_LABEL
        for i, name in self.input_devices:
            if i == self.device:
                return name
        return DEFAULT_MIC_LABEL

    # ---------- UI ----------
    def _build_ui(self):
        self.root.title(f"Bark Detector v{self.version}")
        self.root.geometry("780x700")
        self.root.configure(bg="#1e1e1e")
        label_opts = dict(bg="#1e1e1e", fg="#eeeeee")

        self.status_label = tk.Label(
            self.root, text="Starting...", font=("Segoe UI", 12),
            justify="left", anchor="w", **label_opts,
        )
        self.status_label.pack(fill="x", padx=12, pady=(10, 4))

        self.canvas = tk.Canvas(self.root, width=750, height=230, bg="#111111",
                                 highlightthickness=0)
        self.canvas.pack(padx=12, pady=6)

        self.detail_label = tk.Label(
            self.root, text="", font=("Segoe UI", 10), fg="#9a9a9a",
            justify="left", anchor="w", bg="#1e1e1e",
        )
        self.detail_label.pack(fill="x", padx=12, pady=(0, 6))

        # ---- microphone selector ----
        mic_frame = tk.Frame(self.root, bg="#1e1e1e")
        mic_frame.pack(fill="x", padx=12, pady=(0, 6))
        tk.Label(mic_frame, text="Microphone:", font=("Segoe UI", 10, "bold"),
                 **label_opts).pack(side="left")

        self.mic_values = [DEFAULT_MIC_LABEL] + [f"{i}: {name}" for i, name in self.input_devices]
        self.mic_var = tk.StringVar(value=self._mic_current_value())
        self.mic_combo = ttk.Combobox(
            mic_frame, textvariable=self.mic_var, values=self.mic_values,
            state="readonly", width=60,
        )
        self.mic_combo.pack(side="left", padx=(10, 0), fill="x", expand=True)
        self.mic_combo.bind("<<ComboboxSelected>>", self._on_mic_change)

        # ---- threshold slider ----
        thresh_frame = tk.Frame(self.root, bg="#1e1e1e")
        thresh_frame.pack(fill="x", padx=12, pady=(4, 4))

        tk.Label(thresh_frame, text="Bark threshold:", font=("Segoe UI", 10, "bold"),
                 **label_opts).pack(side="left")
        self.threshold_value_label = tk.Label(
            thresh_frame, text=f"{self.threshold:.2f}", font=("Segoe UI", 10, "bold"),
            fg="#4ea1ff", bg="#1e1e1e", width=5,
        )
        self.threshold_value_label.pack(side="right")

        self.threshold_slider = tk.Scale(
            thresh_frame, from_=0.0, to=1.0, resolution=0.01, orient="horizontal",
            length=460, showvalue=False, bg="#1e1e1e", fg="#eeeeee",
            troughcolor="#333333", highlightthickness=0, activebackground="#4ea1ff",
            command=self._on_threshold_slide,
        )
        self.threshold_slider.set(self.threshold)
        self.threshold_slider.pack(side="left", fill="x", expand=True, padx=(10, 10))

        # ---- cooldown slider ----
        cooldown_frame = tk.Frame(self.root, bg="#1e1e1e")
        cooldown_frame.pack(fill="x", padx=12, pady=(4, 8))

        tk.Label(cooldown_frame, text="Cooldown (s):", font=("Segoe UI", 10, "bold"),
                 **label_opts).pack(side="left")
        self.cooldown_value_label = tk.Label(
            cooldown_frame, text=f"{self.cooldown:.0f}", font=("Segoe UI", 10, "bold"),
            fg="#4ea1ff", bg="#1e1e1e", width=5,
        )
        self.cooldown_value_label.pack(side="right")

        self.cooldown_slider = tk.Scale(
            cooldown_frame, from_=0, to=60, resolution=1, orient="horizontal",
            length=460, showvalue=False, bg="#1e1e1e", fg="#eeeeee",
            troughcolor="#333333", highlightthickness=0, activebackground="#4ea1ff",
            command=self._on_cooldown_slide,
        )
        self.cooldown_slider.set(self.cooldown)
        self.cooldown_slider.pack(side="left", fill="x", expand=True, padx=(10, 10))

        # ---- alert sounds panel ----
        sounds_frame = tk.LabelFrame(
            self.root, text="Alert sounds (tick to enable)", font=("Segoe UI", 10, "bold"),
            bg="#1e1e1e", fg="#eeeeee", labelanchor="nw",
        )
        sounds_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        self.sounds_list_frame = tk.Frame(sounds_frame, bg="#1e1e1e")
        self.sounds_list_frame.pack(fill="both", expand=True, padx=10, pady=(8, 4), anchor="w")

        buttons_row = tk.Frame(sounds_frame, bg="#1e1e1e")
        buttons_row.pack(fill="x", padx=10, pady=(4, 4))

        tk.Button(buttons_row, text="Add sound file(s)...", command=self._add_sound_files).pack(side="left")
        tk.Button(buttons_row, text="Refresh list", command=self._refresh_sound_list).pack(side="left", padx=(8, 0))
        tk.Label(buttons_row, text="(add/remove .wav files in the 'sounds' folder next to the app, then Refresh)",
                 font=("Segoe UI", 8), fg="#777777", bg="#1e1e1e").pack(side="left", padx=(10, 0))

        mode_row = tk.Frame(sounds_frame, bg="#1e1e1e")
        mode_row.pack(fill="x", padx=10, pady=(2, 8))
        tk.Label(mode_row, text="Play ticked sounds:", **label_opts).pack(side="left")
        self.sound_mode_var = tk.StringVar(value=self.sound_mode)
        tk.Radiobutton(mode_row, text="In order", variable=self.sound_mode_var, value="order",
                        command=self._on_mode_change, bg="#1e1e1e", fg="#eeeeee",
                        selectcolor="#333333", activebackground="#1e1e1e").pack(side="left", padx=(8, 0))
        tk.Radiobutton(mode_row, text="Random", variable=self.sound_mode_var, value="random",
                        command=self._on_mode_change, bg="#1e1e1e", fg="#eeeeee",
                        selectcolor="#333333", activebackground="#1e1e1e").pack(side="left", padx=(8, 0))

    def _mic_current_value(self):
        if self.device is None:
            return DEFAULT_MIC_LABEL
        for i, name in self.input_devices:
            if i == self.device:
                return f"{i}: {name}"
        return DEFAULT_MIC_LABEL

    def _on_mic_change(self, _event=None):
        value = self.mic_var.get()
        if value == DEFAULT_MIC_LABEL:
            new_device = None
        else:
            try:
                new_device = int(value.split(":", 1)[0])
            except (ValueError, IndexError):
                new_device = None

        if new_device == self.device:
            return

        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

        self.device = new_device
        self.mic_error = None
        try:
            self._start_audio()
        except Exception as e:
            self.mic_error = str(e)
            messagebox.showerror("Microphone", f"Could not open that microphone:\n{e}")
        self._save_config()

    def _on_threshold_slide(self, value):
        self.threshold = float(value)
        self.threshold_value_label.config(text=f"{self.threshold:.2f}")
        self._save_config()

    def _on_cooldown_slide(self, value):
        self.cooldown = float(value)
        self.cooldown_value_label.config(text=f"{self.cooldown:.0f}")
        self._save_config()

    def _on_mode_change(self):
        self.sound_mode = self.sound_mode_var.get()
        self._play_index = 0
        self._save_config()

    def _refresh_sound_list(self):
        for w in self.sounds_list_frame.winfo_children():
            w.destroy()

        try:
            self.available_sounds = sorted(
                f for f in os.listdir(self.sounds_dir) if f.lower().endswith(".wav")
            )
        except OSError:
            self.available_sounds = []

        previously_enabled = self._get_enabled_sound_names() if self.sound_vars else None
        enabled_default = set(previously_enabled) if previously_enabled is not None else (
            set(self._enabled_from_cfg) if self._enabled_from_cfg else set(self.available_sounds)
        )

        self.sound_vars = {}
        if not self.available_sounds:
            tk.Label(
                self.sounds_list_frame,
                text="No sound files yet - click 'Add sound file(s)...' below.\n"
                     "Until you add one, the built-in beep is used.",
                bg="#1e1e1e", fg="#9a9a9a", font=("Segoe UI", 9), justify="left",
            ).pack(anchor="w")
        else:
            for name in self.available_sounds:
                var = tk.BooleanVar(value=name in enabled_default)
                tk.Checkbutton(
                    self.sounds_list_frame, text=name, variable=var,
                    bg="#1e1e1e", fg="#eeeeee", selectcolor="#333333",
                    activebackground="#1e1e1e", activeforeground="#ffffff",
                    font=("Segoe UI", 9), anchor="w",
                    command=self._on_sound_toggle,
                ).pack(anchor="w", fill="x")
                self.sound_vars[name] = var

        self._save_config()

    def _on_sound_toggle(self):
        self._play_index = 0
        self._save_config()

    def _get_enabled_sound_names(self):
        return [name for name, var in self.sound_vars.items() if var.get()]

    def _add_sound_files(self):
        paths = filedialog.askopenfilenames(
            title="Choose alert sound file(s)", filetypes=[("WAV audio", "*.wav")]
        )
        for p in paths:
            try:
                dest = os.path.join(self.sounds_dir, os.path.basename(p))
                if os.path.abspath(p) != os.path.abspath(dest):
                    shutil.copy(p, dest)
            except Exception as e:
                messagebox.showerror("Add sound file", f"Could not add {p}:\n{e}")
        if paths:
            self._refresh_sound_list()

    # ---------- audio + config persistence ----------
    def _save_config(self):
        path = os.path.join(base_dir(), CONFIG_FILENAME)
        cfg = configparser.ConfigParser()
        data = dict(self._cfg_raw)
        data["bark_threshold"] = f"{self.threshold:.2f}"
        data["cooldown_seconds"] = f"{self.cooldown:.0f}"
        data["device"] = "" if self.device is None else str(self.device)
        data["sound_mode"] = self.sound_mode
        data["sound_files_enabled"] = ",".join(self._get_enabled_sound_names())
        cfg["bark"] = data
        try:
            with open(path, "w") as f:
                cfg.write(f)
        except OSError as e:
            print(f"  ! could not save config.ini: {e}")

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

    def _play_next_sound(self):
        enabled = [os.path.join(self.sounds_dir, n) for n in self._get_enabled_sound_names()]
        if not enabled:
            play_alert("")  # built-in beep fallback
            return
        if self.sound_mode == "random":
            play_alert(random.choice(enabled))
        else:
            play_alert(enabled[self._play_index % len(enabled)])
            self._play_index += 1

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
            self._play_next_sound()

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
            text=f"v{self.version}  mic={self._mic_display_name()}  mode={self.sound_mode}"
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
