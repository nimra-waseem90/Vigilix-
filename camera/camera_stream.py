import cv2


class CameraStream:

    def __init__(self, source=0):

        self.source = source
        self.cap = None

    # ========================================================
    # START STREAM
    # ========================================================

    def start(self):

        print(
            f"\n📡 Connecting to source: "
            f"{self.source}"
        )

        self.cap = cv2.VideoCapture(
            self.source
        )

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Unable to open video source: "
                f"{self.source}"
            )

        print("✅ Stream connected successfully")

        return self

    # ========================================================
    # READ FRAME
    # ========================================================

    def read(self):

        if self.cap is None:

            raise RuntimeError(
                "CameraStream has not been started."
            )

        ret, frame = self.cap.read()

        if not ret:

            return None

        return frame

    # ========================================================
    # GET FPS
    # ========================================================

    def get_fps(self):

        if self.cap is None:

            return 0.0

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:

            fps = 30.0

        return fps

    # ========================================================
    # RELEASE STREAM
    # ========================================================

    def release(self):

        if self.cap is not None:

            self.cap.release()
            self.cap = None

        print("📴 Stream released.")