import cv2

from camera.camera_stream import CameraStream


VIDEO_SOURCE = "data/raw/videos/test.mp4"


def main():

    print("\n========================================")
    print("       VIGILIX CAMERA STREAM TEST")
    print("========================================")

    camera = CameraStream(
        source=VIDEO_SOURCE
    )

    try:

        # Start the stream
        camera.start()

        print("\n✅ Video stream opened successfully")

        # Get FPS
        fps = camera.get_fps()

        print(f"FPS: {fps}")

        print("\nPress Q to stop the stream.")

        frame_count = 0

        while True:

            frame = camera.read()

            # End of video
            if frame is None:

                print("\n⚠️ End of video reached.")
                break

            frame_count += 1

            # Display frame
            cv2.imshow(
                "Vigilix Camera Stream",
                frame
            )

            # Print every 30 frames
            if frame_count % 30 == 0:

                print(
                    f"Frames received: "
                    f"{frame_count}"
                )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):

                print("\n🛑 Stream stopped by user.")
                break

    except Exception as e:

        print("\n❌ Camera stream error:")
        print(e)

    finally:

        camera.release()

        cv2.destroyAllWindows()

        print("\n========================================")
        print("       CAMERA STREAM TEST COMPLETE")
        print("========================================")


if __name__ == "__main__":
    main()