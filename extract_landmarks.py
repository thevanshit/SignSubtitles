"""
Extract hand landmarks using MediaPipe Tasks Python API.
Processed 459 videos → 63 features × 30 frames = landmark-based dataset.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

SEQUENCE_LENGTH = 30
MODEL_PATH = '/tmp/hand_landmarker.task'

PHRASE_MAP = {
    'HELLO': 0,
    'HRU': 1,
    'FINE': 2,
    'YES': 3,
    'NO': 4,
    'HELP': 5,
    'THANK': 6,
    'PLEASE': 7,
    'SLOW': 8,
    'NICE': 9,
}


def create_landmarker():
    """Create MediaPipe hand landmarker."""
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks_from_video(video_path, landmarker, max_frames=SEQUENCE_LENGTH):
    """Extract hand landmarks from video file using a fresh landmarker for each video."""
    # Create fresh landmarker for each video to avoid timestamp issues
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        detector.close()
        return None
    
    # Get video FPS for proper timing
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    sequence = []
    timestamp_ms = 0
    
    while len(sequence) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        results = detector.detect_for_video(mp_image, timestamp_ms)
        
        keypoints = np.zeros(63, dtype=np.float32)
        
        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand = results.hand_landmarks[0]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
            keypoints = coords.flatten()
        
        sequence.append(keypoints)
        timestamp_ms += int(1000 / fps)
    
    cap.release()
    detector.close()
    
    if len(sequence) < SEQUENCE_LENGTH:
        while len(sequence) < SEQUENCE_LENGTH:
            sequence.append(np.zeros(63, dtype=np.float32))
    
    return np.array(sequence[:SEQUENCE_LENGTH], dtype=np.float32)


def main():
    VIDEO_DIR = 'raw_videos/my_recordings'
    OUTPUT_DIR = 'data'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting landmark extraction...")
    print("Using MediaPipe HandLandmarker (Tasks API)")
    
    video_files = list(Path(VIDEO_DIR).glob('*.mov'))
    print(f"Found {len(video_files)} videos")
    
    all_sequences = []
    all_labels = []
    
    for video_file in tqdm(video_files, desc="Extracting landmarks"):
        filename = video_file.stem
        
        phrase = None
        for p in PHRASE_MAP:
            if filename.startswith(p):
                phrase = p
                break
        
        if phrase is None:
            continue
        
        sequence = extract_landmarks_from_video(str(video_file), None)
        
        if sequence is not None:
            all_sequences.append(sequence)
            all_labels.append(PHRASE_MAP[phrase])
    
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Unique labels: {np.unique(y)}")
    print(f"Samples per class:")
    for label_idx, label_name in enumerate(PHRASE_MAP):
        count = np.sum(y == label_idx)
        print(f"  {label_name}: {count}")
    
    np.save(f'{OUTPUT_DIR}/X_landmarks.npy', X)
    np.save(f'{OUTPUT_DIR}/y_landmarks.npy', y)
    
    with open(f'{OUTPUT_DIR}/label_encoder_landmarks.pkl', 'wb') as f:
        pickle.dump(PHRASE_MAP, f)
    
    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  X_landmarks.npy: {X.shape}")
    print(f"  y_landmarks.npy: {y.shape}")


if __name__ == '__main__':
    main()