import cv2

RTSP_URL = "rtsp://127.0.0.1:8554/fire"

print("Connecting to RTSP stream...")

cap = cv2.VideoCapture(RTSP_URL)

print("isOpened:", cap.isOpened())

print("\nTrying to read one frame...")

ret, frame = cap.read()

print("ret =", ret)

if frame is None:
    print("frame = None")
else:
    print("frame received!")
    print("frame shape =", frame.shape)

    cv2.imshow("RTSP Test", frame)
    cv2.waitKey(5000)

cap.release()
cv2.destroyAllWindows()

print("\nTest finished.")