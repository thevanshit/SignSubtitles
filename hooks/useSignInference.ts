'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import { HandLandmarker, PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import metadata from '../public/model/metadata.json';

const SEQUENCE_LENGTH = 30;
const FEATURES_PER_FRAME = 225;
const MIN_FRAMES_FOR_PREDICTION = 30; // Must be exactly 30 for model
const PHRASES = metadata.class_names as string[];

const PREDICTION_WINDOW_SIZE = 10;
const STABILITY_THRESHOLD = 8;
const CONFIDENCE_THRESHOLD = 0.5;
const PREDICTION_COOLDOWN = 800;

export interface SignPrediction {
  phrase: string;
  confidence: number;
}

export interface UseSignInferenceOptions {
  onPrediction?: (prediction: SignPrediction | null) => void;
  minConfidence?: number;
  fps?: number;
}

function getMostFrequent(arr: string[]): string {
  const freq: Record<string, number> = {};
  arr.forEach(v => {
    freq[v] = (freq[v] || 0) + 1;
  });
  return Object.keys(freq).reduce((a, b) =>
    freq[a] > freq[b] ? a : b
  );
}

export function useSignInference(options: UseSignInferenceOptions = {}) {
  const { onPrediction, minConfidence = CONFIDENCE_THRESHOLD, fps = 10 } = options;

  const [isLoading, setIsLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPrediction, setCurrentPrediction] = useState<SignPrediction | null>(null);
  const [handDetected, setHandDetected] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const frameBufferRef = useRef<number[]>([]);
  const rafRef = useRef<number | null>(null);
  const lastProcessRef = useRef<number>(0);
  const processInterval = 1000 / fps;

  const predictionWindowRef = useRef<string[]>([]);
  const stabilityCountRef = useRef(0);
  const lastStablePredictionRef = useRef('');
  const lastPredictionTimeRef = useRef(0);
  const streamRef = useRef<MediaStream | null>(null);

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

      // CHECKPOINT 1: Model loaded
      console.log('✅ CHECKPOINT 1 - Model loaded:', !!modelRef.current);
      console.log('   Model input shape:', modelRef.current.inputs[0].shape);
      console.log('   Classes:', PHRASES);

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

      const poseFeatures: number[] = poseDetectedNow
        ? poseResults.landmarks![0].flatMap((lm) => [lm.x, lm.y, lm.z ?? 0])
        : new Array(99).fill(0);

      let leftHandFeatures: number[] = new Array(63).fill(0);
      if (handResults.landmarks && handResults.landmarks.length > 0 && handResults.handednesses) {
        const leftIdx = handResults.handednesses.findIndex((h) => h[0]?.categoryName === 'Left');
        if (leftIdx >= 0) {
          leftHandFeatures = handResults.landmarks[leftIdx].flatMap((lm) => [lm.x, lm.y, lm.z ?? 0]);
        }
      }

      let rightHandFeatures: number[] = new Array(63).fill(0);
      if (handResults.landmarks && handResults.landmarks.length > 0 && handResults.handednesses) {
        const rightIdx = handResults.handednesses.findIndex((h) => h[0]?.categoryName === 'Right');
        if (rightIdx >= 0) {
          rightHandFeatures = handResults.landmarks[rightIdx].flatMap((lm) => [lm.x, lm.y, lm.z ?? 0]);
        }
      }

      const allFeatures = [...poseFeatures, ...leftHandFeatures, ...rightHandFeatures];

      if (allFeatures.length !== FEATURES_PER_FRAME) {
        console.log('❌ CHECKPOINT 2 - Keypoints length MISMATCH:', allFeatures.length, 'expected:', FEATURES_PER_FRAME);
        return new Array(FEATURES_PER_FRAME).fill(0);
      }

      // Log detection status
      if (handDetectedNow || poseDetectedNow) {
        console.log('📷 Frame:', handDetectedNow ? '👋' : '', poseDetectedNow ? '🏃' : '', '| Features:', allFeatures.length);
      }

      return allFeatures;
    },
    []
  );

  const predict = useCallback(() => {
    const model = modelRef.current;

    // CHECKPOINT 1 (inside predict): Verify model exists
    if (!model) {
      console.log('❌ Model is null!');
      return;
    }

    const bufferLength = frameBufferRef.current.length;
    const minLength = MIN_FRAMES_FOR_PREDICTION * FEATURES_PER_FRAME;

    // CHECKPOINT 3: Buffer fill status (lower threshold for faster response)
    if (bufferLength < minLength) {
      if (bufferLength % 225 === 0) {
        console.log('❌ CHECKPOINT 3 - Buffer filling:', bufferLength, '/', minLength, '(need', MIN_FRAMES_FOR_PREDICTION, 'frames)');
      }
      return;
    }

    console.log('✅ CHECKPOINT 3 - Buffer ready:', bufferLength, '/', minLength, '(' + Math.floor(bufferLength/225) + 'frames)');

    // Use exactly SEQUENCE_LENGTH frames (must be 30 for model)
    const frames = SEQUENCE_LENGTH;
    const featureCount = frames * FEATURES_PER_FRAME;
    
    // Get the last 30 frames from buffer
    const startIdx = Math.max(0, bufferLength - featureCount);
    const sequenceSlice = frameBufferRef.current.slice(startIdx, startIdx + featureCount);
    const sequence = new Float32Array(sequenceSlice);

    console.log('🎬 Processing:', frames, 'frames for prediction (startIdx:', startIdx, ')');

    // FIXED: Use proper tensor creation with expandDims for batch dimension
    // Must be exactly [1, 30, 225]
    const inputTensor = tf.tensor(Array.from(sequence), [frames, FEATURES_PER_FRAME]);
    const input = inputTensor.expandDims(0);

    // CHECKPOINT 4: Input shape verification
    console.log('✅ CHECKPOINT 4 - Input shape:', input.shape);
    console.log('   ===> MUST BE ===> [1, 30, 225]');

    console.log('🔮 Running model.predict()...');
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

    const rawPrediction = PHRASES[maxIdx] || 'UNKNOWN';
    const confidence = maxProb;

    // CHECKPOINT 5: Prediction result with histogram
    const top3 = Array.from(probs)
      .map((p, i) => ({ prob: p, name: PHRASES[i] || 'Class' + i }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 3)
      .map(x => `${x.name}:${(x.prob*100).toFixed(0)}%`);
    
    console.log('✅ CHECKPOINT 5 - Prediction result:');
    console.log('   Index:', maxIdx, '| Word:', rawPrediction, '| Conf:', (confidence * 100).toFixed(1) + '%');
    console.log('   Top 3:', top3.join(' | '));

    input.dispose();
    inputTensor.dispose();
    prediction.dispose();

    const now = Date.now();

    if (confidence >= minConfidence) {
      predictionWindowRef.current.push(rawPrediction);
      if (predictionWindowRef.current.length > PREDICTION_WINDOW_SIZE) {
        predictionWindowRef.current.shift();
      }

      const smoothedPrediction = getMostFrequent(predictionWindowRef.current);

      if (smoothedPrediction === lastStablePredictionRef.current) {
        stabilityCountRef.current++;
      } else {
        stabilityCountRef.current = 1;
        lastStablePredictionRef.current = smoothedPrediction;
      }

      console.log('📊 Stability:', stabilityCountRef.current, '/', STABILITY_THRESHOLD, '| Pred:', smoothedPrediction);

      if (stabilityCountRef.current >= STABILITY_THRESHOLD && now - lastPredictionTimeRef.current > PREDICTION_COOLDOWN) {
        console.log('🎯 FINAL SUBTITLE:', smoothedPrediction, 'Confidence:', (confidence * 100).toFixed(1) + '%');
        const result: SignPrediction = {
          phrase: smoothedPrediction,
          confidence: confidence,
        };
        console.log('📺 SETTING SUBTITLE:', smoothedPrediction);
        setCurrentPrediction(result);
        onPrediction?.(result);
        lastPredictionTimeRef.current = now;
      }
    } else {
      console.log('⚠️ Low confidence:', (confidence * 100).toFixed(1) + '% (threshold:', (minConfidence * 100).toFixed(0) + '%)');
      if (predictionWindowRef.current.length > 0) {
        predictionWindowRef.current.shift();
      }

      if (predictionWindowRef.current.length === 0) {
        stabilityCountRef.current = 0;
        if (lastStablePredictionRef.current !== '') {
          lastStablePredictionRef.current = '';
          setCurrentPrediction(null);
          onPrediction?.(null);
        }
      }
    }
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

      predictionWindowRef.current = [];
      stabilityCountRef.current = 0;
      lastStablePredictionRef.current = '';
      lastPredictionTimeRef.current = 0;

      setIsRunning(true);
      lastProcessRef.current = 0;

      console.log('🎥 Camera started');

      const processFrame = (timestamp: number) => {
        if (timestamp - lastProcessRef.current >= processInterval) {
          lastProcessRef.current = timestamp;

          const features = extractFeatures(timestamp);

          frameBufferRef.current.push(...features);
          const bufLen = frameBufferRef.current.length;
          const maxFrames = SEQUENCE_LENGTH;
          const minFrames = MIN_FRAMES_FOR_PREDICTION;
          
          // Keep buffer manageable (max 30 frames)
          if (bufLen > maxFrames * FEATURES_PER_FRAME) {
            frameBufferRef.current = frameBufferRef.current.slice(-maxFrames * FEATURES_PER_FRAME);
          }

          // Predict when we have minimum frames (now 15 instead of 30)
          if (bufLen >= minFrames * FEATURES_PER_FRAME) {
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

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    frameBufferRef.current = [];
    predictionWindowRef.current = [];
    stabilityCountRef.current = 0;
    lastStablePredictionRef.current = '';
    setIsRunning(false);
    setCurrentPrediction(null);

    console.log('🛑 Camera stopped');
  }, []);

  useEffect(() => {
    let mounted = true;
    const initModel = async () => {
      try {
        await init();
      } catch (err) {
        if (mounted) console.error('Init error:', err);
      }
    };
    initModel();
    return () => {
      mounted = false;
      stopCamera();
      handLandmarkerRef.current?.close();
      poseLandmarkerRef.current?.close();
      modelRef.current?.dispose();
    };
  }, [init, stopCamera]);

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