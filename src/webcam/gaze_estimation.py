
import cv2
import mediapipe as mp
import numpy as np
import joblib
import csv

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"

CALIBRATION_FILE = (
    BASE_DIR / "data" / "calibration" / "calibration_data.csv"
)

TRAINED_MODEL_FILE = (
    BASE_DIR / "data" / "calibration" / "gaze_model.joblib"
)

CAMERA_INDEX = 0

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

FEATURE_COLUMNS = [
    "eye_a_x",
    "eye_a_y",
    "eye_b_x",
    "eye_b_y"
]

TARGET_COLUMNS = [
    "target_x",
    "target_y"
]


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
# 3. EYE FEATURE EXTRACTION
# --------------------------------------------------

def extract_eye_features(landmarks):

    if len(landmarks) < 478:
        return None

    iris_a = landmarks[468]
    iris_b = landmarks[473]

    eye_a_outer = landmarks[33]
    eye_a_inner = landmarks[133]

    eye_b_outer = landmarks[362]
    eye_b_inner = landmarks[263]

    def normalize_eye(iris, corner1, corner2):

        eye_width = abs(
            corner2.x - corner1.x
        )

        if eye_width < 0.001:
            return None

        left_x = min(
            corner1.x,
            corner2.x
        )

        center_y = (
            corner1.y + corner2.y
        ) / 2

        normalized_x = (
            iris.x - left_x
        ) / eye_width

        normalized_y = (
            iris.y - center_y
        ) / eye_width

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
# 4. LOAD CALIBRATION DATA
# --------------------------------------------------

def load_calibration_data():

    if not CALIBRATION_FILE.exists():

        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_FILE}"
        )

    features = []
    targets = []

    with open(
        CALIBRATION_FILE,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = (
            FEATURE_COLUMNS + TARGET_COLUMNS
        )

        if not reader.fieldnames or not all(
            column in reader.fieldnames
            for column in required_columns
        ):

            raise ValueError(
                "Calibration CSV has missing columns."
            )

        for row in reader:

            feature_row = [
                float(row[column])
                for column in FEATURE_COLUMNS
            ]

            target_row = [
                float(row[column])
                for column in TARGET_COLUMNS
            ]

            if not all(
                np.isfinite(value)
                for value in feature_row + target_row
            ):
                continue

            features.append(feature_row)
            targets.append(target_row)

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)

    if len(X) < 20:
        raise ValueError(
            "Not enough valid calibration samples."
        )

    return X, y


# --------------------------------------------------
# 5. TRAIN RANDOM FOREST MODEL
# --------------------------------------------------

def train_gaze_model():

    print("\nLoading calibration data...")

    X, y = load_calibration_data()

    print("Total samples:", len(X))

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest model...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions,
        multioutput="uniform_average"
    )

    print("\n========== MODEL EVALUATION ==========")

    print(f"Test MAE: {mae:.4f}")
    print(f"Test R2 Score: {r2:.4f}")

    # Save model and feature information
    model_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "test_mae": float(mae),
        "test_r2": float(r2)
    }

    TRAINED_MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model_bundle,
        TRAINED_MODEL_FILE
    )

    print("\nModel saved successfully:")
    print(TRAINED_MODEL_FILE)

    print(
        "\nNote: Random sample splitting can produce "
        "optimistic metrics because nearby video frames "
        "may be very similar. Independent-session testing "
        "is still required."
    )

    return model


# --------------------------------------------------
# 6. LIVE GAZE ESTIMATION
# --------------------------------------------------

def run_live_gaze_estimation(model):

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Face Landmarker model missing: {MODEL_PATH}"
        )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1
    )

    detector = FaceLandmarker.create_from_options(
        options
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        camera.release()

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not camera.isOpened():

        detector.close()

        raise RuntimeError(
            "Unable to open webcam."
        )

    gaze_window = "EyeSense AI - Gaze Estimation"
    camera_window = "EyeSense AI - Camera"

    cv2.namedWindow(
        gaze_window,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        gaze_window,
        SCREEN_WIDTH,
        SCREEN_HEIGHT
    )

    cv2.namedWindow(
        camera_window,
        cv2.WINDOW_NORMAL
    )

    print("\nLive gaze estimation started.")
    print("Press Q to exit.")

    frame_count = 0

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("Unable to read camera frame.")
                break

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
                frame_count * 1000 / 30
            )

            frame_count += 1

            results = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # Create gaze display
            screen = np.zeros(
                (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
                dtype=np.uint8
            )

            screen[:] = (25, 25, 25)

            cv2.putText(
                screen,
                "Approximate Gaze Position",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            if results.face_landmarks:

                landmarks = results.face_landmarks[0]

                features = extract_eye_features(
                    landmarks
                )

                if features is not None:

                    predicted_position = model.predict(
                        np.asarray(
                            [features],
                            dtype=np.float64
                        )
                    )[0]

                    raw_x, raw_y = predicted_position

                    # Clamp only for display
                    display_x = np.clip(
                        raw_x,
                        0.0,
                        1.0
                    )

                    display_y = np.clip(
                        raw_y,
                        0.0,
                        1.0
                    )

                    screen_x = int(
                        display_x * (SCREEN_WIDTH - 1)
                    )

                    screen_y = int(
                        display_y * (SCREEN_HEIGHT - 1)
                    )

                    cv2.circle(
                        screen,
                        (screen_x, screen_y),
                        15,
                        (0, 255, 0),
                        -1
                    )

                    cv2.putText(
                        screen,
                        f"X: {raw_x:.3f}  Y: {raw_y:.3f}",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Gaze Estimated",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                else:

                    cv2.putText(
                        screen,
                        "Eye landmarks unavailable",
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

            else:

                cv2.putText(
                    screen,
                    "No Face Detected",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                gaze_window,
                screen
            )

            cv2.imshow(
                camera_window,
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.release()
        detector.close()

        cv2.destroyAllWindows()

        print("Gaze estimation stopped.")


# --------------------------------------------------
# 7. MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    trained_model = train_gaze_model()

    run_live_gaze_estimation(
        trained_model
    )
