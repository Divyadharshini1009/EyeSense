
import cv2
import mediapipe as mp
import numpy as np
import joblib
import csv
import time

from pathlib import Path

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "face_landmarker.task"
)

GAZE_MODEL_FILE = (
    BASE_DIR
    / "data"
    / "calibration"
    / "gaze_model.joblib"
)

OUTPUT_DIR = BASE_DIR / "data" / "gaze"

GAZE_OUTPUT_FILE = OUTPUT_DIR / "gaze_samples.csv"

FIXATION_OUTPUT_FILE = OUTPUT_DIR / "fixations.csv"

CAMERA_INDEX = 0

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# I-VT configuration
VELOCITY_THRESHOLD = 1.0
MIN_FIXATION_DURATION = 0.10

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
# 4. SAVE GAZE SAMPLES
# --------------------------------------------------

def save_gaze_samples(samples):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    headers = [
        "timestamp",
        "gaze_x",
        "gaze_y"
    ]

    with open(
        GAZE_OUTPUT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(headers)

        writer.writerows(samples)

    print("\nGaze samples saved:")
    print(GAZE_OUTPUT_FILE)

# --------------------------------------------------
# 5. I-VT FIXATION DETECTION
# --------------------------------------------------

def detect_fixations(samples):

    if len(samples) < 2:
        print("Not enough gaze samples.")
        return []

    fixations = []

    # Calculate gaze velocity for each sample.
    # Coordinates are normalized screen coordinates.
    velocities = [0.0]

    for i in range(1, len(samples)):

        previous = samples[i - 1]
        current = samples[i]

        time_difference = (
            current[0] - previous[0]
        )

        if time_difference <= 0:
            velocities.append(float("inf"))
            continue

        dx = current[1] - previous[1]
        dy = current[2] - previous[2]

        distance = np.sqrt(
            dx ** 2 + dy ** 2
        )

        velocity = (
            distance / time_difference
        )

        velocities.append(velocity)

    # Identify consecutive low-velocity samples.
    fixation_indices = [
        i
        for i, velocity in enumerate(velocities)
        if velocity <= VELOCITY_THRESHOLD
    ]

    if not fixation_indices:
        return []

    # Group consecutive samples into fixation candidates.
    groups = []

    current_group = [fixation_indices[0]]

    for index in fixation_indices[1:]:

        if index == current_group[-1] + 1:

            current_group.append(index)

        else:

            groups.append(current_group)

            current_group = [index]

    groups.append(current_group)

    # Keep candidates lasting long enough.
    for group in groups:

        start_index = group[0]
        end_index = group[-1]

        start_time = samples[start_index][0]
        end_time = samples[end_index][0]

        duration = end_time - start_time

        if duration < MIN_FIXATION_DURATION:
            continue

        x_positions = [
            samples[i][1]
            for i in group
        ]

        y_positions = [
            samples[i][2]
            for i in group
        ]

        fixation_x = float(
            np.mean(x_positions)
        )

        fixation_y = float(
            np.mean(y_positions)
        )

        fixations.append({
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "gaze_x": fixation_x,
            "gaze_y": fixation_y,
            "sample_count": len(group)
        })

    return fixations

# --------------------------------------------------
# 6. SAVE FIXATION RESULTS
# --------------------------------------------------

def save_fixations(fixations):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    headers = [
        "fixation_id",
        "start_time",
        "end_time",
        "duration_seconds",
        "gaze_x",
        "gaze_y",
        "sample_count"
    ]

    with open(
        FIXATION_OUTPUT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(headers)

        for index, fixation in enumerate(
            fixations,
            start=1
        ):

            writer.writerow([
                index,
                fixation["start_time"],
                fixation["end_time"],
                fixation["duration"],
                fixation["gaze_x"],
                fixation["gaze_y"],
                fixation["sample_count"]
            ])

    print("\nFixation results saved:")
    print(FIXATION_OUTPUT_FILE)

# --------------------------------------------------
# 7. LIVE GAZE COLLECTION
# --------------------------------------------------

def collect_gaze_samples(model):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face model not found: {MODEL_PATH}"
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

    gaze_window = "EyeSense AI - Fixation Collection"

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

    samples = []

    frame_count = 0

    print("\n========== GAZE COLLECTION ==========")
    print("Keep your face in front of the webcam.")
    print("Move your eyes naturally.")
    print("Press Q to stop and extract fixations.")

    start_time = time.monotonic()

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("Unable to read webcam.")
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

            timestamp_ms = frame_count * 33

            frame_count += 1

            results = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )

            screen = np.zeros(
                (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
                dtype=np.uint8
            )

            screen[:] = (25, 25, 25)

            cv2.putText(
                screen,
                "EyeSense AI - Gaze Tracking",
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

                    prediction = model.predict(
                        np.asarray(
                            [features],
                            dtype=np.float64
                        )
                    )[0]

                    gaze_x = float(
                        np.clip(prediction[0], 0.0, 1.0)
                    )

                    gaze_y = float(
                        np.clip(prediction[1], 0.0, 1.0)
                    )

                    timestamp = (
                        time.monotonic() - start_time
                    )

                    samples.append([
                        timestamp,
                        gaze_x,
                        gaze_y
                    ])

                    screen_x = int(
                        gaze_x * (SCREEN_WIDTH - 1)
                    )

                    screen_y = int(
                        gaze_y * (SCREEN_HEIGHT - 1)
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
                        f"X: {gaze_x:.3f} Y: {gaze_y:.3f}",
                        (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Gaze Tracking Active",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
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

            cv2.putText(
                screen,
                f"Samples: {len(samples)}",
                (30, SCREEN_HEIGHT - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
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

    print("\nGaze collection stopped.")

    print(
        "Total gaze samples:",
        len(samples)
    )

    return samples

# --------------------------------------------------
# 8. MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    if not GAZE_MODEL_FILE.exists():

        raise FileNotFoundError(
            "Trained gaze model not found. "
            "Run gaze_estimation.py first."
        )

    print("Loading trained gaze model...")

    model_bundle = joblib.load(
        GAZE_MODEL_FILE
    )

    gaze_model = model_bundle["model"]

    gaze_samples = collect_gaze_samples(
        gaze_model
    )

    if gaze_samples:

        save_gaze_samples(
            gaze_samples
        )

        print("\nRunning I-VT fixation extraction...")

        fixation_results = detect_fixations(
            gaze_samples
        )

        save_fixations(
            fixation_results
        )

        print("\n========== FIXATION SUMMARY ==========")

        print(
            "Total gaze samples:",
            len(gaze_samples)
        )

        print(
            "Detected fixations:",
            len(fixation_results)
        )

        if fixation_results:

            average_duration = np.mean([
                item["duration"]
                for item in fixation_results
            ])

            print(
                f"Average fixation duration: "
                f"{average_duration:.3f} seconds"
            )

        print("\nModule 5 completed.")

    else:

        print(
            "No valid gaze samples collected. "
            "Fixation extraction skipped."
        )