
import cv2
import mediapipe as mp
import numpy as np
import csv
import time

from pathlib import Path


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"

OUTPUT_DIR = BASE_DIR / "data" / "calibration"
OUTPUT_FILE = OUTPUT_DIR / "calibration_data.csv"

WINDOW_NAME = "EyeSense AI - Eye Calibration"
CAMERA_NAME = "EyeSense - Camera Preview"

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

CAMERA_INDEX = 0

TARGET_DURATION = 3.0
COLLECTION_START = 1.0


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
# 3. CALIBRATION TARGETS
# --------------------------------------------------

TARGETS = [
    (0.10, 0.10),
    (0.50, 0.10),
    (0.90, 0.10),

    (0.10, 0.50),
    (0.50, 0.50),
    (0.90, 0.50),

    (0.10, 0.90),
    (0.50, 0.90),
    (0.90, 0.90)
]


# --------------------------------------------------
# 4. EYE LANDMARK FEATURE EXTRACTION
# --------------------------------------------------

def extract_eye_features(landmarks):

    if len(landmarks) < 478:
        return None

    height = 1.0
    width = 1.0

    # Iris centers
    iris_a = landmarks[468]
    iris_b = landmarks[473]

    # Eye corner landmarks
    eye_a_outer = landmarks[33]
    eye_a_inner = landmarks[133]

    eye_b_outer = landmarks[362]
    eye_b_inner = landmarks[263]

    def normalize_eye(iris, corner1, corner2):

        eye_width = abs(corner2.x - corner1.x)

        if eye_width < 0.001:
            return None

        left_x = min(corner1.x, corner2.x)

        center_y = (corner1.y + corner2.y) / 2

        normalized_x = (
            (iris.x - left_x) / eye_width
        )

        normalized_y = (
            (iris.y - center_y) / eye_width
        )

        return normalized_x, normalized_y

    eye_a = normalize_eye(
        iris_a,
        eye_a_outer,
        eye_a_inner
    )

    eye_b = normalize_eye(
        iris_b,
        eye_b_outer,
        eye_b_inner
    )

    if eye_a is None or eye_b is None:
        return None

    return [
        eye_a[0],
        eye_a[1],
        eye_b[0],
        eye_b[1]
    ]


# --------------------------------------------------
# 5. DRAW CALIBRATION TARGET
# --------------------------------------------------

def draw_target(target_x, target_y, elapsed):

    screen = np.zeros(
        (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
        dtype=np.uint8
    )

    screen[:] = (25, 25, 25)

    x = int(target_x * SCREEN_WIDTH)
    y = int(target_y * SCREEN_HEIGHT)

    cv2.circle(
        screen,
        (x, y),
        22,
        (0, 255, 255),
        -1
    )

    cv2.circle(
        screen,
        (x, y),
        32,
        (255, 255, 255),
        2
    )

    cv2.putText(
        screen,
        "Look directly at the yellow target",
        (300, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    remaining = max(
        0,
        int(TARGET_DURATION - elapsed)
    )

    cv2.putText(
        screen,
        f"Please focus: {remaining} seconds",
        (440, 680),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2
    )

    return screen


# --------------------------------------------------
# 6. MAIN CALIBRATION CLASS
# --------------------------------------------------

class EyeCalibration:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODEL_PATH)
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=1
        )

        self.detector = (
            FaceLandmarker.create_from_options(options)
        )

        self.camera = None
        self.samples = []

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------
    # 7. OPEN CAMERA
    # --------------------------------------------------

    def open_camera(self):

        self.camera = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )

        if not self.camera.isOpened():

            self.camera.release()

            self.camera = cv2.VideoCapture(
                CAMERA_INDEX
            )

        if not self.camera.isOpened():

            raise RuntimeError(
                "Unable to open webcam."
            )

        print("Webcam opened successfully.")

    # --------------------------------------------------
    # 8. START CALIBRATION
    # --------------------------------------------------

    def run(self):

        self.open_camera()

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            SCREEN_WIDTH,
            SCREEN_HEIGHT
        )

        cv2.namedWindow(
            CAMERA_NAME,
            cv2.WINDOW_NORMAL
        )

        print("\nEyeSense AI Calibration Started")
        print("Keep your head reasonably still.")
        print("Look at each yellow target.")
        print("Press SPACE to begin.")
        print("Press Q to quit.")

        try:

            while True:

                success, frame = self.camera.read()

                if not success:
                    raise RuntimeError(
                        "Unable to read webcam."
                    )

                frame = cv2.flip(frame, 1)

                cv2.putText(
                    frame,
                    "Press SPACE to start calibration",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(CAMERA_NAME, frame)

                start_screen = np.zeros(
                    (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
                    dtype=np.uint8
                )

                start_screen[:] = (25, 25, 25)

                cv2.putText(
                    start_screen,
                    "EyeSense AI - Eye Calibration",
                    (260, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    start_screen,
                    "Look at each yellow target when it appears.",
                    (260, 320),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2
                )

                cv2.putText(
                    start_screen,
                    "Press SPACE to begin | Q to quit",
                    (350, 400),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(
                    WINDOW_NAME,
                    start_screen
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    return

                if key == 32:
                    break

            # Process all nine targets
            for index, target in enumerate(TARGETS):

                target_x, target_y = target

                print(
                    f"\nCalibration point {index + 1}/9"
                )

                target_start = time.monotonic()

                while True:

                    elapsed = (
                        time.monotonic() - target_start
                    )

                    if elapsed >= TARGET_DURATION:
                        break

                    success, frame = self.camera.read()

                    if not success:
                        continue

                    frame = cv2.flip(frame, 1)

                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame
                    )

                    timestamp_ms = int(
                        time.monotonic() * 1000
                    )

                    results = self.detector.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )

                    # Collect features after the initial settling period
                    if (
                        elapsed >= COLLECTION_START
                        and results.face_landmarks
                    ):

                        landmarks = results.face_landmarks[0]

                        features = extract_eye_features(
                            landmarks
                        )

                        if features is not None:

                            self.samples.append([
                                target_x,
                                target_y,
                                *features
                            ])

                    # Show target
                    target_screen = draw_target(
                        target_x,
                        target_y,
                        elapsed
                    )

                    cv2.imshow(
                        WINDOW_NAME,
                        target_screen
                    )

                    cv2.imshow(
                        CAMERA_NAME,
                        frame
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print("Calibration cancelled.")
                        return

                print(
                    f"Point {index + 1} completed."
                )

            self.save_calibration_data()

        finally:

            if self.camera is not None:
                self.camera.release()

            self.detector.close()

            cv2.destroyAllWindows()

            print("\nCalibration session ended.")

    # --------------------------------------------------
    # 9. SAVE CALIBRATION DATA
    # --------------------------------------------------

    def save_calibration_data(self):

        if not self.samples:

            print(
                "No eye features collected. "
                "Check face detection and try again."
            )

            return

        headers = [
            "target_x",
            "target_y",
            "eye_a_x",
            "eye_a_y",
            "eye_b_x",
            "eye_b_y"
        ]

        with open(
            OUTPUT_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow(headers)

            writer.writerows(self.samples)

        print("\n========== CALIBRATION COMPLETED ==========")

        print(
            f"Collected samples: {len(self.samples)}"
        )

        print(
            f"Saved file: {OUTPUT_FILE}"
        )


# --------------------------------------------------
# 10. PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    calibration = EyeCalibration()

    calibration.run()
