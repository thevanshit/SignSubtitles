# ISHARA - Sign Language Detection & Subtitling

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0097A7?style=flat-square&logo=mediapipe&logoColor=white)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

Real-time sign language recognition system that detects hand gestures from webcam video and displays subtitles. Uses an LSTM deep learning model trained on MediaPipe hand landmarks.

## Overview

ISHARA captures video from a webcam, extracts hand landmark coordinates using MediaPipe, and feeds them into a trained LSTM model to recognize sign language phrases in real time. Detected signs are displayed as subtitles on the interface.

## Features

- **Real-time Inference** — Webcam-based sign language detection using MediaPipe landmarks
- **LSTM Model** — Trained on 10 sign language phrases (HELLO, HRU, FINE, YES, NO, HELP, THANK, PLEASE, SLOW, NICE)
- **Live Subtitles** — Detected signs displayed as on-screen subtitles
- **Hand Detection Indicator** — Visual feedback when hands are detected
- **Modern UI** — Clean interface with camera preview and subtitle history

## Project Structure

```
SignSubtitles/
├── app/                    # Next.js frontend
│   ├── page.tsx           # Main meeting page with camera integration
│   ├── layout.tsx         # App layout
│   └── globals.css        # Global styles
├── hooks/
│   └── useSignInference.ts   # Webcam + model inference hook
├── store/
│   └── meetingStore.ts    # Zustand state management
├── train_model.py         # LSTM model training script
├── extract_landmarks.py   # MediaPipe landmark extraction from videos
├── convert_tfjs.py        # Convert Keras model to TensorFlow.js format
├── test_mediapipe.py      # MediaPipe pipeline test
├── rebuild_model.py       # Model rebuild utility
├── sign_lstm_model.keras  # Trained LSTM model (Keras)
├── sign_model.keras       # Alternative trained model
├── scaler.pkl             # Feature scaler
├── label_encoder.pkl      # Label encoder
├── data/                  # Training data (landmarks)
├── raw_videos/            # Source video files
└── package.json           # Node.js dependencies
```

## Training Pipeline

### 1. Data Collection
Raw sign language videos stored in `raw_videos/`.

### 2. Landmark Extraction
```bash
python extract_landmarks.py
```
Uses MediaPipe Holistic to extract hand landmark coordinates from videos. Outputs normalized landmark sequences to `data/`.

### 3. Model Training
```bash
python train_model.py
```
Trains a Sequential LSTM model with:
- 2 LSTM layers (64 units each) with Dropout
- Dense output layer with Softmax activation
- Early stopping validation
- Train/test split (80/20)

### 4. TF.js Conversion (for browser inference)
```bash
python convert_tfjs.py
```
Converts the trained Keras model to TensorFlow.js format for in-browser inference.

## Running the App

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and grant camera access to start detecting signs.

## Tech Stack

| Component       | Technology                                |
| --------------- | ----------------------------------------- |
| Model Training  | Python, TensorFlow/Keras, LSTM, MediaPipe |
| Frontend        | Next.js 14, TypeScript, Tailwind CSS      |
| Browser ML      | TensorFlow.js, MediaPipe Hand Landmarks   |
| State           | Zustand                                   |
| Icons           | Lucide React                              |

## License

MIT License
