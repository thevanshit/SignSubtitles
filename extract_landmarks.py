import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
import numpy as np
import os
import pickle

model_path = '/tmp/holistic_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading holistic landmarker model...")
    import urllib.request
    url = 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task'
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

options = vision.HolisticLandmarkerOptions(
    base_options=base_options.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    min_pose_landmarks_confidence=0.5,
    min_hand_landmarks_confidence=0.5
)

video_folder = "raw_videos/my_recordings"

data = []
phrases = ["HELLO", "I AM FINE", "HELP", "HOW ARE YOU", "NICE TO MEET YOU",
           "NO", "PLEASE REPEAT", "SLOW DOWN", "THANK YOU", "YES"]

video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
total = len(video_files)
print(f"Starting landmark extraction on {total} videos...")

global_frame_idx = 0

with vision.HolisticLandmarker.create_from_options(options) as landmarker:
    for i, filename in enumerate(video_files):
        lower_name = filename.lower()
        label = None
        
        if "fine" in lower_name:
            label = "I AM FINE"
        elif "hru" in lower_name:
            label = "HOW ARE YOU"
        elif "help" in lower_name:
            label = "HELP"
        elif "nice to meet you" in lower_name:
            label = "NICE TO MEET YOU"
        elif "thank_you" in lower_name or "thank you" in lower_name:
            label = "THANK YOU"
        elif "please_repeat" in lower_name or "please repeat" in lower_name:
            label = "PLEASE REPEAT"
        elif "slow_down" in lower_name or "slow down" in lower_name:
            label = "SLOW DOWN"
        else:
            for p in phrases:
                if p.lower() in lower_name:
                    label = p
                    break
        
        if not label:
            continue
        
        video_path = os.path.join(video_folder, filename)
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        local_frame_count = 0
        skip_counter = 0
        
        while cap.isOpened() and local_frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break
            
            skip_counter += 1
            if skip_counter % 2 == 1:
                frame_small = cv2.resize(frame, (320, 240))
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = landmarker.detect_for_video(mp_image, (global_frame_idx + 1) * 33)
                global_frame_idx += 1
                
                pose_list = results.pose_landmarks
                lh_list = results.left_hand_landmarks
                rh_list = results.right_hand_landmarks
                
                pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_list]).flatten() if pose_list else np.zeros(33*3)
                lh = np.array([[lm.x, lm.y, lm.z] for lm in lh_list]).flatten() if lh_list else np.zeros(21*3)
                rh = np.array([[lm.x, lm.y, lm.z] for lm in rh_list]).flatten() if rh_list else np.zeros(21*3)
                
                landmarks = np.concatenate([pose, lh, rh])
                frames.append(landmarks)
                local_frame_count += 1
        
        cap.release()
        
        while len(frames) < 30:
            frames.append(np.zeros(165))
        
        sequence = np.array(frames[:30])
        data.append({'label': label, 'sequence': sequence})
        
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  Processed {i + 1}/{total} videos ({len(data)} sequences so far)")

print(f"Extracted {len(data)} sequences!")
np.save('train_data.npy', data)
with open('train_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Saved train_data.npy and train_data.pkl")
