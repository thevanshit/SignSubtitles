'use client';

import { useSignInference } from '../hooks/useSignInference';
import { useMeetingStore } from '../store/meetingStore';
import { useEffect, useRef, useCallback } from 'react';
import { Video, VideoOff, Hand, Sparkles } from 'lucide-react';

const PHRASES = [
  'HELLO',
  'HRU',
  'FINE',
  'YES',
  'NO',
  'HELP',
  'THANK',
  'PLEASE',
  'SLOW',
  'NICE',
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
    <main className="min-h-screen bg-white">
      <div className="max-w-3xl mx-auto px-6 py-12">
        {/* Header */}
        <header className="flex items-center justify-between mb-16 animate-fade-in-up">
          <div className="flex items-center gap-3">
            <div className="relative">
              <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="48" height="48" rx="12" fill="#0a0a0a"/>
                <path d="M24 10C24 10 32 15 32 23C32 26.5 30 29.2 27 30.5V34C27 35.7 25.7 37 24 37C22.3 37 21 35.7 21 34V30.5C17.5 29.2 15 26.5 15 23C15 15 24 10 24 10Z" fill="white"/>
                <path d="M18 34V38C18 39.7 19.3 41 21 41H23V36" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M24 37V41H26V37" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M28 34V41H30C31.7 41 33 39.7 33 38V34" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <div className={`absolute -top-1 -right-1 w-3 h-3 rounded-full transition-colors duration-300 ${isRunning ? 'bg-green-500' : 'bg-gray-300'}`} />
            </div>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight" style={{ fontFamily: 'Space Grotesk, system-ui, sans-serif' }}>
                ISHARA
              </h1>
              <p className="text-xs text-gray-400 tracking-wide">Sign Language Recognition</p>
            </div>
          </div>
          
          <div className="flex items-center gap-6">
            <div className={`text-sm transition-colors duration-300 ${isRunning ? 'text-gray-900' : 'text-gray-400'}`}>
              {isRunning ? 'Live' : 'Ready'}
            </div>
            {isLoading && (
              <div className="text-sm text-gray-400 animate-pulse">Loading</div>
            )}
          </div>
        </header>

        {/* Main Video Card */}
        <div className="mb-12 animate-fade-in-up delay-150">
          <div className="relative rounded-2xl overflow-hidden" style={{ boxShadow: '0 4px 40px rgba(0, 0, 0, 0.06), 0 0 0 1px rgba(0, 0, 0, 0.04)' }}>
            {/* Video */}
            <div className="aspect-[4/3] bg-gray-50 relative">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
                style={{ transform: 'scaleX(-1)' }}
              />

              {/* Camera Off State */}
              {!isRunning && !isLoading && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-50">
                  <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                    <Video className="w-8 h-8 text-gray-300" />
                  </div>
                  <p className="text-lg font-medium text-gray-400">Camera Off</p>
                </div>
              )}

              {/* Loading State */}
              {isLoading && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/90">
                  <div className="w-12 h-12 rounded-full border-2 border-gray-200 border-t-gray-800 animate-spin mb-4" />
                  <p className="text-sm font-medium text-gray-600">Loading AI models...</p>
                </div>
              )}

              {/* Running Overlays */}
              {isRunning && (
                <>
                  {/* Status Pills */}
                  <div className="absolute top-4 left-4 right-4 flex justify-between">
                    <div className={`px-4 py-2 rounded-full bg-white/90 backdrop-blur-sm flex items-center gap-2 transition-all duration-300 ${
                      handDetected ? 'ring-1 ring-gray-900' : ''
                    }`}>
                      <Hand className={`w-4 h-4 transition-colors duration-300 ${handDetected ? 'text-gray-900' : 'text-gray-400'}`} />
                      <span className={`text-sm font-medium transition-colors duration-300 ${handDetected ? 'text-gray-900' : 'text-gray-400'}`}>
                        {handDetected ? 'Hand detected' : 'Searching...'}
                      </span>
                    </div>
                    
                    <div className="px-4 py-2 rounded-full bg-white/90 backdrop-blur-sm">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-gray-900 animate-pulse" />
                        <span className="text-sm font-medium text-gray-700">LIVE</span>
                      </div>
                    </div>
                  </div>

                  {/* Prediction Badge */}
                  {currentPrediction && (
                    <div className="absolute bottom-4 right-4 animate-scale-in">
                      <div className="px-5 py-3 rounded-full bg-gray-900 text-white flex items-center gap-3">
                        <Sparkles className="w-4 h-4" />
                        <span className="text-lg font-semibold tracking-tight">{currentPrediction.phrase}</span>
                        <span className="text-xs opacity-70">{Math.round(currentPrediction.confidence * 100)}%</span>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Control Bar */}
            <div className="bg-white px-6 py-4 flex items-center justify-center border-t border-gray-100">
              <button
                onClick={isRunning ? stopCamera : handleStartCamera}
                disabled={isLoading}
                className={`px-8 py-3 rounded-full text-sm font-medium transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed ${
                  isRunning
                    ? 'bg-gray-900 text-white hover:bg-gray-800'
                    : 'bg-gray-900 text-white hover:bg-gray-800'
                }`}
              >
                <span className="flex items-center gap-2">
                  {isRunning ? (
                    <>
                      <VideoOff className="w-4 h-4" />
                      Stop
                    </>
                  ) : (
                    <>
                      <Video className="w-4 h-4" />
                      Start Camera
                    </>
                  )}
                </span>
              </button>
            </div>
          </div>
        </div>

        {/* Subtitle Display */}
        <div className="mb-12 animate-fade-in-up delay-225">
          <div className="py-16 text-center">
            <p className={`text-6xl font-semibold tracking-tight transition-all duration-500 ${
              currentPrediction && isRunning ? 'text-gray-900 animate-pulse' : 'text-gray-200'
            }`} style={{ fontFamily: 'Space Grotesk, system-ui, sans-serif' }}>
              {isRunning ? (
                currentPrediction ? currentPrediction.phrase : 'Show your signs'
              ) : (
                'Start camera to begin'
              )}
            </p>
          </div>
        </div>

        {/* Supported Phrases */}
        <div className="mb-12 animate-fade-in-up delay-300">
          <div className="mb-6">
            <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wider">Supported Phrases</h2>
          </div>
          <div className="flex flex-wrap gap-2">
            {PHRASES.map((phrase, i) => (
              <div
                key={phrase}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 ${
                  currentPrediction?.phrase === phrase && isRunning
                    ? 'bg-gray-900 text-white'
                    : 'bg-gray-50 text-gray-600 hover:bg-gray-100'
                }`}
                style={{ animationDelay: `${i * 30}ms` }}
              >
                {phrase}
              </div>
            ))}
          </div>
        </div>

        {/* Recent Predictions */}
        <div className="animate-fade-in-up delay-375">
          <div className="mb-6 flex items-center justify-between">
            <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wider">Recent Predictions</h2>
            <span className="text-xs text-gray-300">{subtitleHistory.length} detected</span>
          </div>
          
          <div className="bg-gray-50 rounded-2xl p-6">
            {subtitleHistory.length === 0 ? (
              <div className="py-8 text-center">
                <p className="text-gray-400">No predictions yet</p>
                <p className="text-sm text-gray-300 mt-1">Start the camera to begin recognition</p>
              </div>
            ) : (
              <div className="space-y-2">
                {[...subtitleHistory].reverse().slice(0, 6).map((entry, i) => (
                  <div
                    key={i}
                    className={`flex items-center justify-between py-3 px-4 rounded-xl transition-all duration-300 ${
                      i === 0 ? 'bg-white shadow-sm' : ''
                    }`}
                  >
                    <span className="font-medium text-gray-900">{entry.phrase}</span>
                    <div className="flex items-center gap-4 text-sm text-gray-400">
                      <span>{Math.round(entry.confidence * 100)}%</span>
                      <span>{entry.time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-16 pt-8 border-t border-gray-100 text-center">
          <p className="text-xs text-gray-300">Powered by MediaPipe Hands & TensorFlow.js</p>
        </footer>
      </div>

      {/* Error Toast */}
      {error && (
        <div className="fixed bottom-8 left-1/2 -translate-x-1/2 px-6 py-3 rounded-full bg-red-500 text-white text-sm font-medium shadow-lg animate-fade-in">
          {error}
        </div>
      )}
    </main>
  );
}