'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { HandLandmarker, PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import metadata from '../public/model/metadata.json';

const SEQUENCE_LENGTH = 30;
const FEATURES_PER_FRAME = 225;
const PHRASES = metadata.class_names as string[];

// Debouncing config
const STABILITY_THRESHOLD = 15; // Need 15 consecutive same predictions
const PREDICTION_COOLDOWN = 500; // Only show new prediction after 500ms

export interface SignPrediction {
  phrase: string;
  confidence: number;
}

export interface UseSignInferenceOptions {
  onPrediction?: (prediction: SignPrediction | null) => void;
  minConfidence?: number;
  fps?: number;
}

export function useSignInference(options: UseSignInferenceOptions = {}) {
  const { onPrediction, minConfidence = 0.2, fps = 10 } = options;

  const [isLoading, setIsLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPrediction, setCurrentPrediction] = useState<SignPrediction | null>(null);
  const [handDetected, setHandDetected] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoContainerRef = useRef<HTMLDivElement | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const frameBufferRef = useRef<number[]>([]);
  const rafRef = useRef<number | null>(null);
  const lastProcessRef = useRef<number>(0);
  const processInterval = 1000 / fps;

  // Debouncing state
  const predictionCountRef = useRef(0);
  const lastPredictionRef = useRef('');
  const lastPredictionTimeRef = useRef(0);
  const consecutiveNullCountRef = useRef(0);

  const init = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      console.log('Loading MediaPipe and TensorFlow.js models...');

      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );

      handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
      });

      poseLandmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
      });

      modelRef.current = await tf.loadLayersModel('/model/model.json');
      console.log('Model loaded successfully!');
      console.log('Model input shape:', modelRef.current.inputs[0].shape);
      console.log('Classes:', PHRASES);

      setIsLoading(false);
    } catch (err) {
      console.error('Initialization error:', err);
      setError(err instanceof Error ? err.message : 'Failed to initialize models');
      setIsLoading(false);
    }
  }, []);

  const extractFeatures = useCallback(
    (timestamp: number): number[] => {
      const hand = handLandmarkerRef.current;
      const pose = poseLandmarkerRef.current;
      const video = videoRef.current;

      if (!hand || !pose || !video) return new Array(FEATURES_PER_FRAME).fill(0);

      const handResults = hand.detectForVideo(video, timestamp);
      const poseResults = pose.detectForVideo(video, timestamp);

      const handDetectedNow = !!(handResults.landmarks && handResults.landmarks.length > 0);
      const poseDetectedNow = !!(poseResults.landmarks && poseResults.landmarks.length > 0);

      setHandDetected(handDetectedNow);

      // Pose features: 33 landmarks × 3 coordinates = 99 features
      const poseFeatures: number[] = poseDetectedNow
        ? poseResults.landmarks![0].flatMap((lm) => [lm.x, lm.y, lm.z ?? 0])
        : new Array(99).fill(0);

      // Left hand: 21 landmarks × 3 = 63 features
      let leftHandFeatures: number[] = new Array(63).fill(0);
      if (handResults.landmarks && handResults.landmarks.length > 0 && handResults.handednesses) {
        const leftIdx = handResults.handednesses.findIndex((h) => h[0]?.categoryName === 'Left');
        if (leftIdx >= 0) {
          leftHandFeatures = handResults.landmarks[leftIdx].flatMap((lm) => [lm.x, lm.y, lm.z ?? 0]);
        }
      }

      // Right hand: 21 landmarks × 3 = 63 features
      let rightHandFeatures: number[] = new Array(63).fill(0);
      if (handResults.landmarks && handResults.landmarks.length > 0 && handResults.handednesses) {
        const rightIdx = handResults.handednesses.findIndex((h) => h[0]?.categoryName === 'Right');
        if (rightIdx >= 0) {
          rightHandFeatures = handResults.landmarks[rightIdx].flatMap((lm) => [lm.x, lm.y, lm.z ?? 0]);
        }
      }

      const allFeatures = [...poseFeatures, ...leftHandFeatures, ...rightHandFeatures];

      // Debug log every ~2 seconds
      if (timestamp % 2000 < 20) {
        console.log('Features extracted:', {
          poseDetected: poseDetectedNow,
          handDetected: handDetectedNow,
          featureCount: allFeatures.length,
          sampleValues: allFeatures.slice(0, 6)
        });
      }

      return allFeatures;
    },
    []
  );

  const predict = useCallback(() => {
    const model = modelRef.current;
    if (!model || frameBufferRef.current.length < SEQUENCE_LENGTH * FEATURES_PER_FRAME) {
      return;
    }

    const sequence = new Float32Array(SEQUENCE_LENGTH * FEATURES_PER_FRAME);
    for (let i = 0; i < SEQUENCE_LENGTH * FEATURES_PER_FRAME; i++) {
      sequence[i] = frameBufferRef.current[i];
    }

    const input = tf.tensor3d(Array.from(sequence), [1, SEQUENCE_LENGTH, FEATURES_PER_FRAME]);
    const prediction = model.predict(input) as tf.Tensor;
    const probs = prediction.dataSync();

    // Find max probability
    let maxIdx = 0;
    let maxProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxIdx = i;
      }
    }

    // Log top 3 predictions
    const top3 = Array.from(probs)
      .map((p, i) => ({ phrase: PHRASES[i], prob: p }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 3);
    console.log('Predictions:', top3.map((t) => `${t.phrase}: ${(t.prob * 100).toFixed(1)}%`).join(', '));

    input.dispose();
    prediction.dispose();

    const now = Date.now();
    const predictedPhrase = PHRASES[maxIdx];

    // Debouncing logic
    if (maxProb >= minConfidence) {
      if (predictedPhrase === lastPredictionRef.current) {
        predictionCountRef.current++;
        consecutiveNullCountRef.current = 0;
      } else {
        predictionCountRef.current = 1;
        lastPredictionRef.current = predictedPhrase;
        consecutiveNullCountRef.current = 0;
      }

      // Only show prediction after stable for N consecutive frames
      if (predictionCountRef.current >= STABILITY_THRESHOLD) {
        if (now - lastPredictionTimeRef.current > PREDICTION_COOLDOWN) {
          const result: SignPrediction = {
            phrase: predictedPhrase,
            confidence: maxProb,
          };
          setCurrentPrediction(result);
          onPrediction?.(result);
          lastPredictionTimeRef.current = now;
        }
      }
    } else {
      consecutiveNullCountRef.current++;
      if (consecutiveNullCountRef.current >= STABILITY_THRESHOLD) {
        predictionCountRef.current = 0;
        lastPredictionRef.current = '';
        setCurrentPrediction(null);
        onPrediction?.(null);
      }
    }
  }, [minConfidence, onPrediction]);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      });

      // Create video element
      const video = document.createElement('video');
      video.srcObject = stream;
      video.autoplay = true;
      video.muted = true;
      video.playsInline = true;
      video.style.width = '100%';
      video.style.height = '100%';
      video.style.objectFit = 'cover';
      video.style.transform = 'scaleX(-1)'; // Mirror for selfie view

      videoRef.current = video;

      // Wait for video to be ready
      await new Promise<void>((resolve) => {
        video.onloadedmetadata = () => resolve();
      });
      await video.play();

      // Reset debouncing state
      predictionCountRef.current = 0;
      lastPredictionRef.current = '';
      lastPredictionTimeRef.current = 0;
      consecutiveNullCountRef.current = 0;

      setIsRunning(true);
      lastProcessRef.current = 0;

      const processFrame = (timestamp: number) => {
        if (timestamp - lastProcessRef.current >= processInterval) {
          lastProcessRef.current = timestamp;

          const features = extractFeatures(timestamp);

          frameBufferRef.current.push(...features);
          if (frameBufferRef.current.length > SEQUENCE_LENGTH * FEATURES_PER_FRAME) {
            frameBufferRef.current = frameBufferRef.current.slice(-SEQUENCE_LENGTH * FEATURES_PER_FRAME);
          }

          if (frameBufferRef.current.length >= SEQUENCE_LENGTH * FEATURES_PER_FRAME) {
            predict();
          }
        }

        rafRef.current = requestAnimationFrame(processFrame);
      };

      rafRef.current = requestAnimationFrame(processFrame);
    } catch (err) {
      console.error('Camera error:', err);
      setError('Failed to access camera');
    }
  }, [extractFeatures, predict, processInterval]);

  const stopCamera = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    frameBufferRef.current = [];
    predictionCountRef.current = 0;
    lastPredictionRef.current = '';
    setIsRunning(false);
    setCurrentPrediction(null);
  }, []);

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    let mounted = true;
    init().catch((err) => {
      if (mounted) console.error('Init error:', err);
    });
    return () => {
      mounted = false;
      stopCamera();
      handLandmarkerRef.current?.close();
      poseLandmarkerRef.current?.close();
      modelRef.current?.dispose();
    };
  }, [init, stopCamera]);
  /* eslint-enable react-hooks/set-state-in-effect */

  return {
    isLoading,
    isRunning,
    error,
    currentPrediction,
    handDetected,
    startCamera,
    stopCamera,
    videoRef,
  };
}