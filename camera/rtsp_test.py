import cv2

RTSP_URL = "rtsp://9627b0bf2a7b.entrypoint.cloud.wowza.com:1935/app-p5260J38/66abe4b9_stream1"

print("Connecting to RTSP stream...")

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("❌ Failed to open RTSP stream")
    exit()

print("✅ RTSP stream opened successfully")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PEROP_FRAME_HEIGHT))

print(f"FPS: {fps}")
print(f"Resolution: {width}x{height}")
print()
print("Press Q to stop the stream.")

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to receive frame")
        break

    frame_count += 1

    if frame_count % 30 == 0:
        print(f"Frames received: {frame_count}")

    cv2.imshow("Vigilix RTSP Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stream stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()

print()
print("========================================")
print("       RTSP STREAM TEST COMPLETE")
print("========================================")