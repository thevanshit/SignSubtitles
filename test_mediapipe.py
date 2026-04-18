import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
from urllib.request import urlretrieve

print("=" * 50)
print("SignSubtitle - MediaPipe Test")
print("=" * 50)

model_path = '/tmp/hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        model_path
    )
    print("Model downloaded!")

BaseOptions = python.BaseOptions
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot access webcam")
    exit(1)

print("Camera opened successfully!")
print("Show your hand to the camera.")

has_window = True
try:
    cv2.namedWindow("Test")
except Exception as e:
    print(f"No display available: {e}")
    has_window = False

print("Press 'Q' in the video window to quit." if has_window else "Running headless test mode.")
print("=" * 50)

frame_count = 0
with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame from webcam")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)

        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            print(f" Hand detected! (frame {frame_count})", end="\r")
        else:
            print(f"No hand detected... (frame {frame_count})        ", end="\r")

        frame_count += 1

        if has_window:
            cv2.imshow("SignSubtitle - MediaPipe Test", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        if frame_count >= 30:
            break

cap.release()
if has_window:
    cv2.destroyAllWindows()
print("\n" + "=" * 50)
print("Test finished successfully!")
print(f"Processed {frame_count} frames")
print("=" * 50)