from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo26n.pt")

# Run detection on a test image
results = model(
    "https://ultralytics.com/images/bus.jpg",
    save=True
)

print("Detection completed!")