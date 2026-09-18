import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
from ultralytics import YOLO
from camera.camera_stream import CameraStream
from alert.alert_manager import fire_alert
from datetime import datetime
MODEL_PATH = "model/detection/fire_smoke_model.pt"

CONFIDENCE_THRESHOLD = 0.60

REQUIRED_FRAMES = 3

FIRE_CLASS_ID = 1

RTSP_URL = "rtsp://127.0.0.1:8554/fire"


os.makedirs(
    "alert/screenshots",
    exist_ok=True
)

# LOAD FIRE MODEL

print("Loading Fire YOLO model...")

model = YOLO(MODEL_PATH)

print("Fire model loaded successfully.")

 

print("Starting RTSP camera stream...")

camera = CameraStream(
    source=RTSP_URL
)

camera.start()

print("✅ RTSP stream started successfully.")
print("Press Q to quit.")
consecutive_fire_frames = 0

alert_sent = False

while True:

    frame = camera.read()

    if frame is None:

        print("⚠️ No frame received.")

        continue


    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        device="cpu",
        verbose=False
    )


    fire_detected = False

    highest_confidence = 0.0


    # --------------------------------------------------------
    # CHECK DETECTIONS
    # --------------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )


            # ------------------------------------------------
            # FIRE DETECTED
            # ------------------------------------------------

            if class_id == FIRE_CLASS_ID:

                fire_detected = True

                if confidence > highest_confidence:

                    highest_confidence = confidence


                # Bounding box

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"FIRE {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )


    # ========================================================
    # 3-FRAME CONFIRMATION
    # ========================================================

    if fire_detected:

        consecutive_fire_frames += 1

        print(
            f"🔥 FIRE DETECTED | "
            f"Confirmation: "
            f"{consecutive_fire_frames}/"
            f"{REQUIRED_FRAMES} | "
            f"Confidence: "
            f"{highest_confidence:.2f}"
        )

    else:

        # Reset confirmation if fire disappears

        consecutive_fire_frames = 0


    # ========================================================
    # CONFIRMED FIRE
    # ========================================================

    if (
        consecutive_fire_frames >= REQUIRED_FRAMES
        and not alert_sent
    ):

        print(
            "\n🚨 FIRE CONFIRMED!"
        )


        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )


        # ----------------------------------------------------
        # SCREENSHOT PATH
        # ----------------------------------------------------

        screenshot_path = (
            f"alert/screenshots/"
            f"fire_{timestamp}.jpg"
        )


        # ----------------------------------------------------
        # SAVE SCREENSHOT
        # ----------------------------------------------------

        cv2.imwrite(
            screenshot_path,
            frame
        )


        print(
            f"📸 Fire screenshot saved: "
            f"{screenshot_path}"
        )


        # ----------------------------------------------------
        # TRIGGER FIRE ALARM
        # ----------------------------------------------------

        fire_alert(
            highest_confidence,
            screenshot_path
        )


        print(
            "🚨 Fire alarm triggered."
        )


        # Prevent repeated alerts

        alert_sent = True


    # ========================================================
    # DISPLAY STATUS
    # ========================================================

    if fire_detected:

        cv2.putText(
            frame,
            f"FIRE DETECTION "
            f"{consecutive_fire_frames}/"
            f"{REQUIRED_FRAMES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

    else:

        cv2.putText(
            frame,
            "Monitoring...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    cv2.imshow(
        "Vigilix - Fire Detection",
        frame
    )
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

camera.release()

cv2.destroyAllWindows()

print(
    "Fire RTSP detection stopped."
)
