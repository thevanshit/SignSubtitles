import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

print("=" * 50)
print("SignSubtitle - MediaPipe Test")
print("=" * 50)
print("Camera opened successfully!" if cap.isOpened() else "ERROR: Cannot access webcam")
print("Show your hand to the camera.")
print("Press 'Q' in the video window to quit.")
print("=" * 50)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame from webcam")
        break

    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.left_hand_landmarks or results.right_hand_landmarks:
        print(" Hand detected!", end="\r")
    else:
        print("No hand detected...        ", end="\r")

    cv2.imshow("SignSubtitle - MediaPipe Test", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\n" + "=" * 50)
print("Test finished successfully!")
print("=" * 50)
