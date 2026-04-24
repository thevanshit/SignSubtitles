'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import metadata from '../public/model/metadata.json';

const SEQUENCE_LENGTH = 30;
const MIN_FRAMES_FOR_PREDICTION = 30;
const PHRASES = metadata.class_names as string[];

const PREDICTION_WINDOW_SIZE = 10;
const MIN_CONFIDENCE = 0.5;
const SCALER_MEAN = metadata.scaler_mean as number[];
const SCALER_STD = metadata.scaler_std as number[];

interface SignPrediction {
  phrase: string;
  confidence: number;
}

export interface UseSignInferenceOptions {
  onPrediction?: (prediction: SignPrediction | null) => void;
  minConfidence?: number;
  fps?: number;
}

export function useSignInference(options: UseSignInferenceOptions = {}) {
  const { onPrediction, minConfidence = MIN_CONFIDENCE, fps = 10 } = options;

  const [isLoading, setIsLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPrediction, setCurrentPrediction] = useState<SignPrediction | null>(null);
  const [handDetected, setHandDetected] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const frameBufferRef = useRef<number[]>([]);
  const rafRef = useRef<number | null>(null);
  const lastProcessRef = useRef<number>(0);
  const processInterval = 1000 / fps;

  const predictionHistoryRef = useRef<{ idx: number; conf: number }[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const frameCountRef = useRef(0);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const lastTimestampRef = useRef<number>(0);

  const extractLandmarks = useCallback((video: HTMLVideoElement, timestamp: number): number[] => {
    const landmarker = handLandmarkerRef.current;
    if (!landmarker) {
      console.log('Landmarker not initialized');
      return new Array(63).fill(0);
    }

    try {
      const results = landmarker.detectForVideo(video, timestamp);
      if (results.landmarks && results.landmarks.length > 0) {
        const hand = results.landmarks[0];
        const coords = hand.flatMap((lm) => [lm.x, lm.y, lm.z]);
        setHandDetected(true);
        return coords;
      }
    } catch (e) {
      console.error('Landmark error:', e);
    }

    setHandDetected(false);
    return new Array(63).fill(0);
  }, []);

  const normalizeFeatures = useCallback((features: number[]): number[] => {
    const normalized: number[] = [];
    for (let i = 0; i < 63; i++) {
      const val = (features[i] - (SCALER_MEAN[i] || 0)) / (SCALER_STD[i] || 1);
      normalized.push(val);
    }
    return normalized;
  }, []);

  const predict = useCallback(async () => {
    const model = modelRef.current;
    if (!model) return;

    const bufferLength = frameBufferRef.current.length;
    const minLength = MIN_FRAMES_FOR_PREDICTION * 63;

    if (bufferLength < minLength) return;

    const featureCount = SEQUENCE_LENGTH * 63;
    const startIdx = Math.max(0, bufferLength - featureCount);
    const sequenceSlice = frameBufferRef.current.slice(startIdx, startIdx + featureCount);

    if (sequenceSlice.length < featureCount) return;

    const inputTensor = tf.tensor2d(Array.from(sequenceSlice), [SEQUENCE_LENGTH, 63]);
    const input = inputTensor.expandDims(0);

    const prediction = model.predict(input) as tf.Tensor;
    const probs = new Float32Array(prediction.dataSync());

    let maxIdx = 0;
    let maxProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxIdx = i;
      }
    }

    const confidence = maxProb;

    if (confidence > minConfidence) {
      predictionHistoryRef.current.push({ idx: maxIdx, conf: confidence });
    }

    if (predictionHistoryRef.current.length > PREDICTION_WINDOW_SIZE) {
      predictionHistoryRef.current.shift();
    }

    const history = predictionHistoryRef.current;
    if (history.length < 3) return;

    const freq: Record<number, number> = {};
    let maxCount = 0;
    let smoothedIdx = history[0].idx;

    for (const h of history) {
      freq[h.idx] = (freq[h.idx] || 0) + 1;
      if (freq[h.idx] > maxCount) {
        maxCount = freq[h.idx];
        smoothedIdx = h.idx;
      }
    }

    if (maxCount >= 3 && smoothedIdx >= 0 && smoothedIdx < PHRASES.length) {
      const phrase = PHRASES[smoothedIdx];
      console.log('Predicted:', phrase, 'conf:', confidence.toFixed(2));
      setCurrentPrediction({ phrase, confidence });
      onPrediction?.({ phrase, confidence });
    }

    input.dispose();
    inputTensor.dispose();
    prediction.dispose();
  }, [minConfidence, onPrediction]);

  const startCamera = useCallback(async (videoElement: HTMLVideoElement) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      });

      videoRef.current = videoElement;
      videoElement.srcObject = stream;
      streamRef.current = stream;

      await new Promise<void>((resolve) => {
        videoElement.onloadedmetadata = () => resolve();
      });
      await videoElement.play();

      console.log('Initializing MediaPipe HandLandmarker...');
      
      const wasmFileset = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
      
      const landmarker = await HandLandmarker.createFromOptions(wasmFileset, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numHands: 1
      });
      
      handLandmarkerRef.current = landmarker;
      console.log('MediaPipe HandLandmarker initialized!');

      predictionHistoryRef.current = [];
      frameBufferRef.current = [];
      frameCountRef.current = 0;
      lastTimestampRef.current = 0;
      setIsRunning(true);
      lastProcessRef.current = 0;

      console.log('Camera started, processing landmarks...');

      const processFrame = async (timestamp: number) => {
        if (timestamp - lastProcessRef.current >= processInterval) {
          lastProcessRef.current = timestamp;
          frameCountRef.current += 1;

          if (videoRef.current) {
            const features = extractLandmarks(videoRef.current, timestamp);
            const normalized = normalizeFeatures(features);
            frameBufferRef.current.push(...normalized);

            const maxBufLen = SEQUENCE_LENGTH * 63;
            if (frameBufferRef.current.length > maxBufLen) {
              frameBufferRef.current = frameBufferRef.current.slice(-maxBufLen);
            }

            if (frameBufferRef.current.length >= MIN_FRAMES_FOR_PREDICTION * 63) {
              await predict();
            }
          }
        }

        rafRef.current = requestAnimationFrame(processFrame);
      };

      rafRef.current = requestAnimationFrame(processFrame);
    } catch (err) {
      console.error('Camera error:', err);
      setError(err instanceof Error ? err.message : 'Failed to access camera');
    }
  }, [extractLandmarks, normalizeFeatures, predict, processInterval]);

  const stopCamera = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (handLandmarkerRef.current) {
      handLandmarkerRef.current.close();
      handLandmarkerRef.current = null;
    }

    frameBufferRef.current = [];
    predictionHistoryRef.current = [];
    frameCountRef.current = 0;
    setIsRunning(false);
    setCurrentPrediction(null);
    setHandDetected(false);

    console.log('Camera stopped');
  }, []);

  useEffect(() => {
    let mounted = true;
    const init = async () => {
      try {
        console.log('Loading TensorFlow.js model...');
        const model = await tf.loadLayersModel('/model/tfjs/model.json');
        console.log('Model loaded:', model.inputs[0].shape);
        modelRef.current = model;
        console.log('Model loaded! Classes:', PHRASES);
        setIsLoading(false);
      } catch (err) {
        console.error('Init error:', err);
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Failed to initialize');
        }
        setIsLoading(false);
      }
    };
    init();

    return () => {
      mounted = false;
    };
  }, []);

  return {
    isLoading,
    isRunning,
    error,
    currentPrediction,
    handDetected,
    startCamera,
    stopCamera,
  };
}