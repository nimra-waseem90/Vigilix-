from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.predict(
    source="data/raw/videos/test.mp4",
    save=True,
    conf=0.5
)

print("Video processing completed!")