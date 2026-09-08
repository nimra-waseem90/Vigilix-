from ultralytics import YOLO
from alert.alert_manager import weapon_alert
from datetime import datetime
import cv2
import os

model = YOLO("model/detection/gun_knife_yolo11n.pt")

source = "data/raw/videos/weapon_test.mp4"

confidence_threshold = 0.40
required_frames = 3

consecutive_weapon_frames = 0
alert_sent = False

os.makedirs("alert/screenshots", exist_ok=True)

results = model.predict(
    source=source,
    save=True,
    conf=confidence_threshold,
    stream=True
)

for result in results:

    weapon_detected = False
    highest_confidence = 0.0
    detected_weapon = None

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        weapon_type = result.names[class_id]

        if confidence > highest_confidence:
            highest_confidence = confidence
            detected_weapon = weapon_type

        weapon_detected = True

    if weapon_detected:

        consecutive_weapon_frames += 1
        print(
            f"Weapon detected | "
            f"Frame confirmation: "
            f"{consecutive_weapon_frames}/{required_frames} | "
            f"Type: {detected_weapon} | "
            f"Confidence: {highest_confidence:.2f}"
        )

    else:

        consecutive_weapon_frames = 0

    if (
        consecutive_weapon_frames >= required_frames
        and not alert_sent
    ):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        screenshot_path = (
            f"alert/screenshots/"
            f"weapon_{detected_weapon}_{timestamp}.jpg"
        )

        annotated_image = result.plot()

        cv2.imwrite(
            screenshot_path,
            annotated_image
        )

        weapon_alert(
            detected_weapon,
            highest_confidence,
            screenshot_path
        )

        alert_sent = True

print("Multi-frame weapon detection completed!")