# Bark Detector (ML edition)

A portable Windows app that listens to your microphone, uses Google's
YAMNet sound-classification model to check specifically for a dog
bark/dog/howl/growl sound, and plays an alert sound when it's confident
enough. Shows a live graph of that confidence against your threshold.

Runs fully offline for detection. The only network call is a read-only
check against this repo's GitHub releases, used for self-update.

## Download

Grab `BarkDetectorML.exe` from the
[latest release](https://github.com/duncanbiscuits/bark-detector/releases/latest).
No install, no Python needed - just run it. It's portable, so you can
copy it (and its `config.ini` once created) to any Windows PC or a USB
drive.

## How it works

- Microphone audio is checked ~5x/second against YAMNet's Dog / Bark /
  Bow-wow / Howl / Growling classes.
- If the confidence crosses `bark_threshold` in `config.ini`, it plays
  an alert sound (a built-in beep, or your own `.wav` if you set one)
  and waits `cooldown_seconds` before it can alert again.
- A live graph shows the confidence line and the threshold line so you
  can see it working and tune the threshold visually.

See `config.ini` (auto-created next to the exe on first run) for all
settings.

## Auto-update

On startup, the app checks this repo's latest GitHub release. If a
newer version is published:

1. It downloads the new `BarkDetectorML.exe` from the release assets.
2. It verifies its SHA-256 checksum against `checksums.txt` published
   in the same release. If the checksum doesn't match (or is missing),
   the update is discarded and the app keeps running its current
   version - it will never run an unverified binary.
3. Once verified, it swaps the exe and restarts itself on the new
   version.

Set `auto_update = false` in `config.ini` to disable this and stay on
whatever version you currently have.

## Building it yourself / releasing a new version

Pushing a tag like `v1.2.0` triggers `.github/workflows/release.yml`,
which builds `BarkDetectorML.exe` on a Windows runner, generates
`checksums.txt`, and publishes both as a GitHub Release with that tag
as the version. Any app already installed will pick it up automatically
on its next launch.

To build locally:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\pip.exe install -r requirements.txt
"v0.0.0-local" | Out-File -Encoding ascii -NoNewline version.txt
.\venv\Scripts\pyinstaller.exe --onefile --console --name BarkDetectorML `
  --add-data "yamnet.tflite;." `
  --add-data "yamnet_class_map.csv;." `
  --add-data "version.txt;." `
  --collect-all ai_edge_litert `
  --collect-data certifi `
  bark_detector_ml.py
```

## Security notes

- No telemetry, no data collection, no third-party analytics.
- Detection is fully offline; audio is never recorded or saved.
- Auto-update is read-only against GitHub's public release API and
  will only install a binary whose SHA-256 checksum matches the
  published `checksums.txt` for that release.

## Changelog
- v1.0.1: test release to verify auto-update.

