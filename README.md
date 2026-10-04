# Vigilix — AI Video Surveillance Console

Vigilix watches an RTSP camera stream in real time and raises alerts when it
detects **fire/smoke** or **weapons** (gun/knife), using the YOLO models in
`model/detection/`. This repo now includes a backend that connects the
detection engine to the dashboard UI, so you can paste in an RTSP URL from
the browser and detection starts immediately.

## What's in this update

- **`server/`** — a Flask backend (`server/app.py`, `server/stream_worker.py`,
  `server/store.py`) that wraps the existing `camera/`, `detection/`,
  `fusion/` and `alert/` code. It exposes a login endpoint, lets you
  register an RTSP stream, runs fire + weapon detection on it in a
  background thread, streams the annotated video back to the browser
  (MJPEG), and persists alerts + camera configs to a local SQLite database
  (`data/vigilix.db`) so nothing is lost on restart — the app starts
  completely empty and only ever shows data it actually detected.
- **`frontend/index.html`** — the dashboard UI, fully wired to that backend
  and stripped of mock/placeholder content:
  - Real login (`/api/login`), backed by the `VIGILIX_ADMIN_USER` /
    `VIGILIX_ADMIN_PASS` you set in `.env`.
  - **Fixed:** the login screen no longer leaves the dashboard scrolled out
    of view (it now locks page scroll while open and resets scroll on
    sign-in).
  - **Fixed:** the bell icon used to pop open a fake "new detection" toast
    on click. It now just opens the Logs page; the toast (with the
    built-in alarm sound) fires on its own, automatically, the moment a
    new CRITICAL/HIGH alert actually comes in from a camera.
  - **Cameras/Live** pages → "+ Add camera (RTSP)" connects a real stream
    and starts detection immediately.
  - **Dashboard, Logs, Database, Settings** all read real data from the
    backend (`/api/streams`, `/api/alerts`, `/api/config`) instead of the
    original hardcoded mock rows/numbers. Screens with no backing feature
    yet (sign-in history, operator acknowledge/escalate workflow) say so
    honestly instead of showing fake rows.
  - Removed the unreachable mobile-preview mockup and the "design
    prototype" chrome.
  - The built-in alarm (`alert/alarm.wav`) plays in the browser on every
    new critical/high alert, respecting the Alarm on/off toggle in the
    top bar.

## 1. Requirements

- Python 3.10–3.12 (the bundled models were tested with `ultralytics`; very
  new Python versions may not have wheels for every dependency yet)
- An RTSP-capable camera or stream URL, e.g. `rtsp://user:pass@192.168.1.50:554/stream1`
- FFmpeg (OpenCV uses it to decode RTSP) — install via your OS package
  manager if `cv2.VideoCapture` fails to open a stream:
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: install FFmpeg and add it to PATH

## 2. Install

```bash
cd Vigilix--main
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> GPU note: `requirements.txt` installs plain CPU `torch`. If you have an
> NVIDIA GPU and want faster inference, install the matching CUDA build of
> `torch`/`torchvision` from https://pytorch.org first, then run the
> `pip install -r requirements.txt` above (pip will skip torch if it's
> already satisfied).

## 3. Configure

```bash
cp .env.example .env
```

Edit `.env` and set a real username/password/secret key — these protect
the dashboard.

## 4. Run

```bash
python server/app.py
```

Open **http://localhost:5000** in your browser and sign in with the
credentials from `.env`.

## 5. Add a camera

1. Go to the **Cameras** page (or **Live**) and click **+ Add camera (RTSP)**.
2. Enter a name and the RTSP URL, e.g. `rtsp://192.168.1.50:554/stream1`.
   (To test without a real camera, you can stream one of the sample videos
   in `data/raw/videos/` through a local RTSP server such as
   [MediaMTX](https://github.com/bluenviron/mediamtx) and point Vigilix at
   `rtsp://127.0.0.1:8554/<name>`.)
3. Click **Connect & start detection**. The camera tile goes live within a
   few seconds, and any confirmed fire or weapon detection (3 consecutive
   frames, same thresholds as the original CLI scripts) appears in
   **Logs → Alert log**, with a saved screenshot in `alert/screenshots/`
   and a row in `alert/alert_log.csv` / `alert/weapon_log.csv`.

## Project structure (new pieces only)

```
server/
  app.py            Flask app: auth, /api/streams, /api/streams/<id>/mjpeg, /api/alerts
  stream_worker.py  Per-camera thread: RTSP capture -> fire+weapon YOLO -> fused alerts
frontend/
  index.html         Dashboard UI (served by the Flask app at "/")
.env.example
```

The original detection scripts (`detection/fire_detection.py`,
`detection/weapon_detection.py`, `main.py`) still work standalone if you
want to run detection from the command line instead of the dashboard.

## Troubleshooting

- **"Unable to open video source"** — check the RTSP URL works in VLC
  first (`Media → Open Network Stream`). Most IP cameras need
  `rtsp://user:pass@ip:554/...`; the exact path varies by brand.
- **Stream stuck on "Connecting"** — the camera may be behind NAT/firewall,
  or only accept one RTSP client at a time (close VLC/other viewers).
- **Detection is slow** — the CPU build of `torch` runs both models on
  every frame; expect a few FPS on a laptop CPU. Use a GPU build of
  torch/torchvision for real-time speed.
