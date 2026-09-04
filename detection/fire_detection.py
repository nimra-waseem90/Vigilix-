from ultralytics import YOLO
from alert.alert_manager import fire_alert


model = YOLO("model/detection/fire_smoke_model.pt")

confidence_threshold = 0.60
required_frames = 3

consecutive_fire_frames = 0
alert_sent = False

results = model.predict(
    source="data/raw/videos/fire_test.mp4",
    save=True,
    conf=confidence_threshold,
    imgsz=320,
    device="cpu",
    stream=True,
    verbose=False
)

for frame_number, result in enumerate(results):

    if frame_number % 3 != 0:
        continue

    fire_detected = False
    highest_confidence = 0.0

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if class_id == 1:
            fire_detected = True

            if confidence > highest_confidence:
                highest_confidence = confidence

    if fire_detected:
        consecutive_fire_frames += 1

        print(
            f"🔥 Fire detected | "
            f"Confidence: {highest_confidence:.2f} | "
            f"Frame: {frame_number}"
        )

    else:
        consecutive_fire_frames = 0

    if consecutive_fire_frames >= required_frames and not alert_sent:
        annotated_frame = result.plot() 
        fire_alert(
         highest_confidence,
         annotated_frame
        ) 

        alert_sent = True

print("\nFire/Smoke video detection completed!")
