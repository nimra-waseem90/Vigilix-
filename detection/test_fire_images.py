from ultralytics import YOLO
import cv2
import os

# Load fire and smoke model
model = YOLO("model/detection/fire_smoke_model.pt")

# Test images folder
image_folder = "data/raw/fire_test"

# Output folder
output_folder = "data/raw/results"
os.makedirs(output_folder, exist_ok=True)

# Test all images
for filename in os.listdir(image_folder):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image_path = os.path.join(image_folder, filename)

        print(f"\nTesting: {filename}")

        # Run detection
        results = model(image_path, conf=0.25)

        # Get detection results
        result = results[0]

        # Print detected objects
        if len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                print(
                    f"🔥 {class_name} detected | "
                    f"Confidence: {confidence:.2f}"
                )
        else:
            print("❌ No fire/smoke detected")

        # Draw bounding boxes
        annotated_image = result.plot()

        # Save result
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, annotated_image)

        # Display result
        cv2.imshow("Fire Detection Test", annotated_image)

        key = cv2.waitKey(0)

        if key == ord("q"):
            break

cv2.destroyAllWindows()