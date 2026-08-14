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
copy it (and its `config.ini` and `sounds` folder once created) to any
Windows PC or a USB drive.

## How it works

- Microphone audio is checked ~5x/second against YAMNet's Dog / Bark /
  Bow-wow / Howl / Growling classes.
- If the confidence crosses the threshold, it plays an alert sound and
  waits `cooldown_seconds` before it can alert again.
- A live graph shows the confidence line and the threshold line so you
  can see it working.
- Ships with two default alert sounds (`sit-down.wav` is ticked by
  default, `sit-down-sorry-dog.wav` is available but unticked) so it
  works immediately after download. Both are Barnaby Joyce saying
  "Sit down!" from his [ABC 7.30 Report interview](https://www.youtube.com/shorts/4qlDBmeiGaM).

### Live controls (in the app window, no restart needed)

- **Microphone** - pick any input device from the dropdown; it
  switches immediately.
- **Bark threshold slider** - drag it while the app is running to tune
  sensitivity in real time; the graph's threshold line moves with it.
- **Cooldown slider** - how many seconds to wait between alerts (0-60s).
- **Alert sounds** - drop `.wav` files into the `sounds` folder next
  to the app, click "Refresh list", then tick the ones you want active.
- **Play mode** - "In order" cycles through your ticked sounds one
  after another each time a bark is detected; "Random" picks a
  different ticked one at random each time.

All of the above save to `config.ini` immediately as you change them -
no restart, no manual editing required.

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

Pushing a tag like `v1.3.1` triggers `.github/workflows/release.yml`,
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
  --add-data "default_sounds;default_sounds" `
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
- v1.3.0: microphone picker, cooldown slider, bundled default alert
  sounds (`sit-down.wav` ticked by default, `sit-down-sorry-dog.wav`
  available unticked) - both clipped from Barnaby Joyce's
  [ABC 7.30 Report interview](https://www.youtube.com/shorts/4qlDBmeiGaM)
  saying "Sit down!".
- v1.1.0: live threshold slider, tick-box alert sound list (in
  order / random playback).
- v1.0.1: test release to verify auto-update.
