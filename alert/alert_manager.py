"""
Central alert manager for Vigilix.

Fixes vs. the original version:
- winsound.Beep() only exists on Windows -> crashed / silently failed on
  macOS and Linux. We now use `simpleaudio` to actually play the bundled
  alarm.wav on every OS, with a safe no-op fallback if audio output isn't
  available (e.g. headless server / CI).
- All paths were relative to whatever directory you happened to launch the
  script from (e.g. "alert/alert_log.csv"), which breaks the moment
  Vigilix is imported/run from somewhere else. Paths are now anchored to
  the project root using __file__.
- Added audio_alert() for the new gunshot/scream/glass-break audio model,
  using the same log + screenshot/soundclip + siren pattern as fire/weapon.
"""

from datetime import datetime
import csv
import os

import cv2

# ---------------------------------------------------------------------------
# Paths are anchored to the project root (parent of this "alert" folder),
# so it no longer matters what directory you run a script from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALERT_DIR = os.path.join(PROJECT_ROOT, "alert")
SCREENSHOT_DIR = os.path.join(ALERT_DIR, "screenshots")
ALARM_FILE = os.path.join(ALERT_DIR, "alarm.wav")

FIRE_LOG = os.path.join(ALERT_DIR, "alert_log.csv")
WEAPON_LOG = os.path.join(ALERT_DIR, "weapon_log.csv")
AUDIO_LOG = os.path.join(ALERT_DIR, "audio_log.csv")

os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def _to_relative(path):
    """Store/print paths relative to the project root instead of an
    absolute machine-specific path, so logs stay portable if the project
    folder is moved, zipped, or opened from another machine."""
    if not path:
        return path
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return path  # e.g. different drive letter on Windows


def _play_alarm():
    """Play alert/alarm.wav on any OS. Never raises - a failed beep should
    never crash a security pipeline."""
    try:
        if not os.path.isfile(ALARM_FILE) or os.path.getsize(ALARM_FILE) == 0:
            print("Alarm skipped: alert/alarm.wav is missing or empty. "
                  "Drop a real .wav file there to enable the siren.")
            return
        import simpleaudio as sa
        wave_obj = sa.WaveObject.from_wave_file(ALARM_FILE)
        wave_obj.play()  # non-blocking, so detection loop isn't stalled
    except Exception as e:
        print(f"Alarm could not be played: {e}")


def _write_row(log_file, header, row):
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def fire_alert(confidence, frame=None):
    timestamp = datetime.now()
    timestamp_text = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    filename_time = timestamp.strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 50)
    print("FIRE ALERT")
    print("=" * 50)
    print(f"Time       : {timestamp_text}")
    print(f"Confidence : {confidence:.2f}")
    print("Status     : FIRE CONFIRMED")
    print("=" * 50)

    screenshot_path = ""
    if frame is not None:
        screenshot_path = os.path.join(SCREENSHOT_DIR, f"fire_{filename_time}.jpg")
        cv2.imwrite(screenshot_path, frame)
        print(f"Screenshot saved: {_to_relative(screenshot_path)}")

    _write_row(
        FIRE_LOG,
        ["Date", "Time", "Confidence", "Status", "Screenshot"],
        [timestamp.strftime("%Y-%m-%d"), timestamp.strftime("%H:%M:%S"),
         f"{confidence:.2f}", "FIRE CONFIRMED", _to_relative(screenshot_path)],
    )
    print("Alert logged successfully")
    _play_alarm()


def weapon_alert(weapon_type, confidence, screenshot_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    relative_path = _to_relative(screenshot_path)

    _write_row(
        WEAPON_LOG,
        ["Timestamp", "Event", "Weapon", "Confidence", "Screenshot"],
        [timestamp, "WEAPON DETECTED", weapon_type, f"{confidence:.2f}", relative_path],
    )

    print("\n" + "=" * 50)
    print("WEAPON ALERT")
    print("=" * 50)
    print(f"Time       : {timestamp}")
    print(f"Weapon     : {weapon_type}")
    print(f"Confidence : {confidence:.2f}")
    print(f"Screenshot : {relative_path}")
    print("Status     : WEAPON CONFIRMED")
    print("=" * 50 + "\n")
    _play_alarm()


def audio_alert(label, confidence, clip_path=""):
    """Alert for the audio-detection model (gunshot, scream, glass break,
    explosion, siren, etc.)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    relative_clip_path = _to_relative(clip_path)

    _write_row(
        AUDIO_LOG,
        ["Timestamp", "Event", "Sound", "Confidence", "Clip"],
        [timestamp, "AUDIO EVENT DETECTED", label, f"{confidence:.2f}", relative_clip_path],
    )

    print("\n" + "=" * 50)
    print("AUDIO ALERT")
    print("=" * 50)
    print(f"Time       : {timestamp}")
    print(f"Sound      : {label}")
    print(f"Confidence : {confidence:.2f}")
    if relative_clip_path:
        print(f"Clip       : {relative_clip_path}")
    print("Status     : SOUND CONFIRMED")
    print("=" * 50 + "\n")
    _play_alarm()
