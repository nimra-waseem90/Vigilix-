from fusion.event_format import create_event
from ultralytics import YOLO
from alert.alert_manager import weapon_alert
from datetime import datetime
import cv2
import os


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/detection/gun_knife_yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.40
REQUIRED_FRAMES = 3


# ============================================================
# WEAPON DETECTOR
# ============================================================

def detect_weapon(source):

    model = YOLO(MODEL_PATH)

    consecutive_weapon_frames = 0
    alert_sent = False

    os.makedirs(
        "alert/screenshots",
        exist_ok=True
    )

    results = model.predict(
        source=source,
        save=True,
        conf=CONFIDENCE_THRESHOLD,
        stream=True
    )

    for frame_number, result in enumerate(results):

        weapon_detected = False
        highest_confidence = 0.0
        detected_weapon = None

        # ----------------------------------------------------
        # CHECK DETECTIONS
        # ----------------------------------------------------

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            weapon_type = result.names[class_id]

            if confidence > highest_confidence:

                highest_confidence = confidence
                detected_weapon = weapon_type

            weapon_detected = True

        # ----------------------------------------------------
        # WEAPON DETECTED
        # ----------------------------------------------------

        if weapon_detected:

            consecutive_weapon_frames += 1

            print(
                f"🔫 Weapon detected | "
                f"Frame confirmation: "
                f"{consecutive_weapon_frames}/"
                f"{REQUIRED_FRAMES} | "
                f"Type: {detected_weapon} | "
                f"Confidence: "
                f"{highest_confidence:.2f}"
            )

        else:

            consecutive_weapon_frames = 0

        # ----------------------------------------------------
        # CONFIRMED WEAPON
        # ----------------------------------------------------

        if (
            consecutive_weapon_frames >= REQUIRED_FRAMES
            and not alert_sent
        ):

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            screenshot_path = (
                f"alert/screenshots/"
                f"weapon_{detected_weapon}_"
                f"{timestamp}.jpg"
            )

            annotated_image = result.plot()

            cv2.imwrite(
                screenshot_path,
                annotated_image
            )

            # Existing alert system
            weapon_alert(
                detected_weapon,
                highest_confidence,
                screenshot_path
            )

            # ------------------------------------------------
            # STANDARDIZED EVENT
            # ------------------------------------------------

            weapon_event = create_event(
                event="weapon",
                label=detected_weapon,
                confidence=highest_confidence,
                timestamp=frame_number/30.0
            )

            print(
                "\nSTANDARDIZED WEAPON EVENT:"
            )

            print(weapon_event)

            alert_sent = True

            # ------------------------------------------------
            # RETURN EVENT TO FUSION
            # ------------------------------------------------

            return weapon_event

    print(
        "\nWeapon detection completed!"
    )

    return None


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    event = detect_weapon(
        "data/raw/videos/weapon_test.mp4"
    )

    print(
        "\nFINAL WEAPON EVENT:"
    )

    print(event)