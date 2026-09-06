from datetime import datetime
import csv
import os
import cv2
import winsound


LOG_FILE = "alert/alert_log.csv"
SCREENSHOT_DIR = "alert/screenshots"


def fire_alert(confidence, frame=None):

    timestamp = datetime.now()
    timestamp_text = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    filename_time = timestamp.strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 50)
    print("🚨 FIRE ALERT")
    print("=" * 50)
    print(f"Time       : {timestamp_text}")
    print(f"Confidence : {confidence:.2f}")
    print("Status     : FIRE CONFIRMED")
    print("=" * 50)

    # Create screenshot folder
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # Save screenshot
    screenshot_path = ""

    if frame is not None:

        screenshot_path = os.path.join(
            SCREENSHOT_DIR,
            f"fire_{filename_time}.jpg"
        )

        cv2.imwrite(screenshot_path, frame)

        print(f"📸 Screenshot saved: {screenshot_path}")

    # Save alert log
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Date",
                "Time",
                "Confidence",
                "Status",
                "Screenshot"
            ])

        writer.writerow([
            timestamp.strftime("%Y-%m-%d"),
            timestamp.strftime("%H:%M:%S"),
            f"{confidence:.2f}",
            "FIRE CONFIRMED",
            screenshot_path
        ])

    print("📝 Alert logged successfully")

    # Play alarm
    try:
        winsound.Beep(1000, 700)
        winsound.Beep(1200, 700)
        winsound.Beep(1000, 700)
        print("🔊 Alarm played")

    except Exception as e:
        print(f"⚠️ Alarm could not be played: {e}")

def weapon_alert(weapon_type, confidence, screenshot_path):
    from datetime import datetime
    import csv
    import os

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_file = "alert/weapon_log.csv"

    file_exists = os.path.exists(log_file)

    with open(log_file, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Event",
                "Weapon",
                "Confidence",
                "Screenshot"
            ])

        writer.writerow([
            timestamp,
            "WEAPON DETECTED",
            weapon_type,
            f"{confidence:.2f}",
            screenshot_path
        ])

    print("\n" + "=" * 50)
    print("🚨 WEAPON ALERT")
    print("=" * 50)
    print(f"Time       : {timestamp}")
    print(f"Weapon     : {weapon_type}")
    print(f"Confidence : {confidence:.2f}")
    print(f"Screenshot : {screenshot_path}")
    print("Status     : WEAPON CONFIRMED")
    print("=" * 50 + "\n")

