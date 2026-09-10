from fusion.event_format import create_event
from ultralytics import YOLO
from alert.alert_manager import fire_alert


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/detection/fire_smoke_model.pt"

CONFIDENCE_THRESHOLD = 0.60
REQUIRED_FRAMES = 3
FRAME_SKIP = 3


# ============================================================
# FIRE DETECTOR
# ============================================================

def detect_fire(source):

    model = YOLO(MODEL_PATH)

    consecutive_fire_frames = 0
    alert_sent = False

    results = model.predict(
        source=source,
        save=True,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=320,
        device="cpu",
        stream=True,
        verbose=False
    )

    for frame_number, result in enumerate(results):

        # --------------------------------------------
        # Skip frames for CPU performance
        # --------------------------------------------

        if frame_number % FRAME_SKIP != 0:
            continue

        fire_detected = False
        highest_confidence = 0.0

        # --------------------------------------------
        # Check detections
        # --------------------------------------------

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id == 1:

                fire_detected = True

                if confidence > highest_confidence:
                    highest_confidence = confidence

        # --------------------------------------------
        # Fire detected
        # --------------------------------------------

        if fire_detected:

            consecutive_fire_frames += 1

            print(
                f"🔥 Fire detected | "
                f"Confidence: {highest_confidence:.2f} | "
                f"Frame: {frame_number} | "
                f"Confirmation: "
                f"{consecutive_fire_frames}/{REQUIRED_FRAMES}"
            )

        else:

            consecutive_fire_frames = 0

        # --------------------------------------------
        # CONFIRMED FIRE
        # --------------------------------------------

        if (
            consecutive_fire_frames >= REQUIRED_FRAMES
            and not alert_sent
        ):

            annotated_frame = result.plot()

            # Existing alert system
            fire_alert(
                highest_confidence,
                annotated_frame
            )

            # ----------------------------------------
            # STANDARDIZED EVENT
            # ----------------------------------------

            fire_event = create_event(
                event="fire",
                label="Fire",
                confidence=highest_confidence,
                timestamp=frame_number/30.0
            )

            print(
                "\nSTANDARDIZED FIRE EVENT:"
            )

            print(fire_event)

            alert_sent = True

            # ----------------------------------------
            # RETURN EVENT TO FUSION
            # ----------------------------------------

            return fire_event

    print(
        "\nFire/Smoke video detection completed!"
    )

    return None


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    event = detect_fire(
        "data/raw/videos/fire_test.mp4"
    )

    print(
        "\nFINAL FIRE EVENT:"
    )

    print(event)