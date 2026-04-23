'use client';

import { useSignInference } from '../hooks/useSignInference';
import { useMeetingStore } from '../store/meetingStore';
import { useEffect, useRef, useCallback } from 'react';
import { Video, VideoOff, HandMetal } from 'lucide-react';

const PHRASES = [
  'HELLO',
  'HELP',
  'HOW ARE YOU',
  'I AM FINE',
  'NO',
  'PLEASE REPEAT',
  'SLOW DOWN',
  'THANK YOU',
  'YES',
];

export default function MeetingPage() {
  const subtitleHistory = useMeetingStore((s) => s.subtitleHistory);
  const setPrediction = useMeetingStore((s) => s.setPrediction);
  const videoRef = useRef<HTMLVideoElement>(null);

  const { isLoading, isRunning, error, currentPrediction, handDetected, startCamera, stopCamera } =
    useSignInference({
      onPrediction: setPrediction,
    });

  const handleStartCamera = useCallback(async () => {
    if (videoRef.current) {
      await startCamera(videoRef.current);
    }
  }, [startCamera]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return (
    <main className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
      <div className="w-full max-w-4xl">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          <div className="p-6 border-b border-gray-100">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center">
                  <span className="text-white font-bold text-lg">S</span>
                </div>
                <h1 className="text-xl font-bold text-gray-900">ISHARA</h1>
              </div>
              <div className="flex items-center gap-4">
                {isLoading && (
                  <span className="text-sm text-blue-600 animate-pulse">Loading AI...</span>
                )}
                <div className="flex items-center gap-2">
                  <div
                    className={`w-2.5 h-2.5 rounded-full ${
                      isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-300'
                    }`}
                  />
                  <span className="text-sm text-gray-500 font-medium">
                    {isRunning ? 'Live' : 'Offline'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="p-6">
            <div className="aspect-video bg-black rounded-xl overflow-hidden relative mb-6">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
                style={{ transform: 'scaleX(-1)' }}
              />

              {!isRunning && !isLoading && (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400">
                  <Video className="w-16 h-16 mb-3 opacity-30" />
                  <p className="text-lg font-medium">Camera Off</p>
                  <p className="text-sm text-gray-400 mt-1">Click Start to begin</p>
                </div>
              )}

              {isLoading && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60">
                  <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4" />
                  <p className="text-blue-300 font-medium">Downloading AI models...</p>
                  <p className="text-gray-400 text-sm mt-1">This may take a moment</p>
                </div>
              )}

              {isRunning && (
                <div className="absolute bottom-4 left-4 flex gap-2">
                  <div
                    className={`px-3 py-1.5 rounded-full text-sm font-medium flex items-center gap-1.5 backdrop-blur-sm ${
                      handDetected
                        ? 'bg-green-600/90 text-white'
                        : 'bg-gray-700/80 text-gray-300'
                    }`}
                  >
                    <HandMetal className="w-4 h-4" />
                    {handDetected ? 'Hand detected' : 'No hand'}
                  </div>
                </div>
              )}

              {currentPrediction && isRunning && (
                <div className="absolute bottom-4 right-4">
                  <div className="px-4 py-2 rounded-xl text-lg font-bold bg-blue-600/95 text-white shadow-lg backdrop-blur-sm">
                    {currentPrediction.phrase}
                  </div>
                </div>
              )}
            </div>

            <div className="bg-gray-50 rounded-xl p-4 mb-6 min-h-[80px] flex items-center justify-center">
              <p className={`text-2xl font-bold text-center ${currentPrediction ? 'text-gray-900 animate-pulse' : 'text-gray-400'}`}>
                {isRunning ? (
                  currentPrediction ? currentPrediction.phrase : 'Sign to see subtitles...'
                ) : (
                  'Camera is off'
                )}
              </p>
            </div>

            <div className="flex justify-center mb-6">
              <button
                onClick={isRunning ? stopCamera : handleStartCamera}
                disabled={isLoading}
                className={`px-8 py-3 rounded-xl font-semibold text-lg transition-all flex items-center gap-2 shadow-lg ${
                  isRunning
                    ? 'bg-red-500 hover:bg-red-600 text-white'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {isRunning ? (
                  <>
                    <VideoOff className="w-5 h-5" />
                    Stop Camera
                  </>
                ) : (
                  <>
                    <Video className="w-5 h-5" />
                    Start Camera
                  </>
                )}
              </button>
            </div>

            <div className="border-t border-gray-100 pt-6">
              <h3 className="text-sm font-semibold text-gray-500 mb-3">Supported Phrases</h3>
              <div className="grid grid-cols-3 gap-2 mb-6">
                {PHRASES.map((phrase) => (
                  <div
                    key={phrase}
                    className={`px-3 py-2 rounded-lg text-center text-sm font-medium transition-all ${
                      currentPrediction?.phrase === phrase
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    {phrase}
                  </div>
                ))}
              </div>

              <div>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold text-gray-500">Recent Predictions</h3>
                  <span className="text-xs text-gray-400">{subtitleHistory.length} detected</span>
                </div>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {subtitleHistory.length === 0 ? (
                    <p className="text-gray-400 text-sm text-center py-4">
                      No predictions yet
                    </p>
                  ) : (
                    [...subtitleHistory].reverse().slice(0, 10).map((entry, i) => (
                      <div key={i} className="bg-gray-50 rounded-lg px-4 py-2 flex items-center justify-between">
                        <p className="text-gray-900 font-medium">{entry.phrase}</p>
                        <p className="text-gray-400 text-xs">
                          {(entry.confidence * 100).toFixed(0)}% · {entry.time.toLocaleTimeString()}
                        </p>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-red-500 text-white px-6 py-3 rounded-xl shadow-lg flex items-center gap-2">
          <span className="text-sm font-medium">Error: {error}</span>
        </div>
      )}
    </main>
  );
}