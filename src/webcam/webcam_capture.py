
import cv2
from pathlib import Path
from datetime import datetime


class WebcamCapture:

    def __init__(self, camera_index=0):
        self.camera_index = camera_index

        self.project_root = Path(__file__).resolve().parents[2]

        self.output_folder = self.project_root / "data" / "webcam"
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def start_camera(self):

        self.camera = cv2.VideoCapture(self.camera_index)

        if not self.camera.isOpened():
            print("ERROR: Webcam not detected.")
            return False

        print("EyeSense AI webcam started successfully!")
        return True

    def capture_video(self):

        if not self.start_camera():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_folder / f"webcam_{timestamp}.avi"

        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = self.camera.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 20.0

        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        writer = cv2.VideoWriter(
            str(filepath), fourcc, fps, (width, height)
        )

        if not writer.isOpened():
            print("ERROR: Unable to save video.")
            self.camera.release()
            return

        print("Recording started. Press Q to stop.")

        try:
            while True:

                success, frame = self.camera.read()

                if not success:
                    print("Cannot read webcam frame.")
                    break

                cv2.putText(
                    frame,
                    "EyeSense AI",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                writer.write(frame)

                cv2.imshow("EyeSense AI Webcam", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            writer.release()
            self.camera.release()
            cv2.destroyAllWindows()

        print("Recording completed!")
        print("Saved at:", filepath)


if __name__ == "__main__":
    webcam = WebcamCapture()
    webcam.capture_video()
