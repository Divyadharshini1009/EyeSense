
import cv2
import mediapipe as mp
import numpy as np

from pathlib import Path


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "face_landmarker.task"
)

CAMERA_INDEX = 0


# --------------------------------------------------
# 2. MEDIAPIPE CONFIGURATION
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


# --------------------------------------------------
# 3. EYE LANDMARK INDICES
# --------------------------------------------------

LEFT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]


# --------------------------------------------------
# 4. FACE AND EYE DETECTION CLASS
# --------------------------------------------------

class FaceEyeDetection:

    def __init__(self, camera_index=CAMERA_INDEX):

        self.camera_index = camera_index

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found: {MODEL_PATH}"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = (
            FaceLandmarker.create_from_options(options)
        )

        self.frame_count = 0
        self.fps = 30.0

        print("Face and Eye Detection initialized.")


    # --------------------------------------------------
    # 5. DRAW EYE LANDMARKS
    # --------------------------------------------------

    def draw_eye_landmarks(
        self,
        frame,
        landmarks,
        eye_indices,
        eye_name
    ):

        height, width = frame.shape[:2]

        points = []

        for index in eye_indices:

            landmark = landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append([x, y])

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

        points = np.array(
            points,
            dtype=np.int32
        )

        # Draw eye contour
        cv2.polylines(
            frame,
            [points],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2
        )

        # Display eye name
        first_point = tuple(points[0])

        cv2.putText(
            frame,
            eye_name,
            (first_point[0], first_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )


    # --------------------------------------------------
    # 6. PROCESS FACE LANDMARKS
    # --------------------------------------------------

    def process_face(self, frame, results):

        if not results.face_landmarks:

            cv2.putText(
                frame,
                "No Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            return

        for landmarks in results.face_landmarks:

            # Draw left eye
            self.draw_eye_landmarks(
                frame,
                landmarks,
                LEFT_EYE_INDICES,
                "Left Eye"
            )

            # Draw right eye
            self.draw_eye_landmarks(
                frame,
                landmarks,
                RIGHT_EYE_INDICES,
                "Right Eye"
            )

        cv2.putText(
            frame,
            "Face Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # --------------------------------------------------
    # 7. START WEBCAM DETECTION
    # --------------------------------------------------

    def run(self):

        camera = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW
        )

        if not camera.isOpened():

            camera.release()

            camera = cv2.VideoCapture(
                self.camera_index
            )

        if not camera.isOpened():

            print("ERROR: Cannot access webcam.")

            self.detector.close()
            return

        actual_fps = camera.get(cv2.CAP_PROP_FPS)

        if actual_fps > 0:
            self.fps = actual_fps

        print("Webcam started successfully!")
        print("Face and eye detection running.")
        print("Press Q to exit.")

        try:

            while True:

                success, frame = camera.read()

                if not success:

                    print("Unable to read webcam frame.")
                    break

                # Mirror webcam frame
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Convert to MediaPipe image
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # Generate video timestamp
                timestamp_ms = int(
                    self.frame_count * 1000 / self.fps
                )

                self.frame_count += 1

                # Detect face landmarks
                results = self.detector.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                # Process detected landmarks
                self.process_face(
                    frame,
                    results
                )

                # Display output
                cv2.imshow(
                    "EyeSense AI - Face and Eye Detection",
                    frame
                )

                # Exit on Q
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as error:

            print("Detection error:", error)

        finally:

            camera.release()

            self.detector.close()

            cv2.destroyAllWindows()

            print("Webcam closed.")
            print("Detection completed.")


# --------------------------------------------------
# 8. MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    detector = FaceEyeDetection()

    detector.run()
