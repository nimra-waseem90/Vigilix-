import os

# Force OpenCV to use TCP for RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
from ultralytics import YOLO
from alert.alert_manager import weapon_alert
from fusion.event_format import create_event
from datetime import datetime

from camera.camera_stream import CameraStream


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/detection/gun_knife_yolo11n.pt"

RTSP_URL = "rtsp://127.0.0.1:8554/weapon"

CONFIDENCE_THRESHOLD = 0.40
REQUIRED_FRAMES = 3


# ============================================================
# WEAPON DETECTOR
# ============================================================

def detect_weapon(camera):

    model = YOLO(MODEL_PATH)

    consecutive_weapon_frames = 0
    alert_sent = False

    os.makedirs(
        "alert/screenshots",
        exist_ok=True
    )

    frame_number = 0

    print("\n🔫 Weapon detector started.")
    print("Press Q to quit.\n")

    while True:

        frame = camera.read()

        if frame is None:

            print("⚠️ Failed to read frame.")

            break

        frame_number += 1

        # ----------------------------------------------------
        # RUN YOLO ON CURRENT FRAME
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]

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
        # DRAW DETECTIONS
        # ----------------------------------------------------

        annotated_frame = result.plot()

        cv2.imshow(
            "Vigilix - Weapon Detection",
            annotated_frame
        )

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

            cv2.imwrite(
                screenshot_path,
                annotated_frame
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
                timestamp=frame_number / camera.get_fps()
            )

            print(
                "\n🚨 STANDARDIZED WEAPON EVENT:"
            )

            print(weapon_event)

            alert_sent = True

            # ------------------------------------------------
            # RETURN EVENT TO FUSION
            # ------------------------------------------------

            return weapon_event

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            print("\nWeapon detection stopped by user.")

            break

    cv2.destroyAllWindows()

    print(
        "\nWeapon detection completed!"
    )

    return None


# ============================================================
# STANDALONE RTSP TEST
# ============================================================

if __name__ == "__main__":

    camera = CameraStream(
        source=RTSP_URL
    )

    try:

        camera.start()

        event = detect_weapon(camera)

        print(
            "\nFINAL WEAPON EVENT:"
        )

        print(event)

    finally:

        camera.release()