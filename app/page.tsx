'use client';

import { useSignInference } from '../hooks/useSignInference';
import { useMeetingStore } from '../store/meetingStore';
import { useEffect, useRef } from 'react';
import { Video, VideoOff, HandMetal, User } from 'lucide-react';

const PHRASES = [
  'HELLO',
  'HOW ARE YOU',
  'I AM FINE',
  'THANK YOU',
  'HELP',
  'YES',
  'NO',
  'PLEASE REPEAT',
  'SLOW DOWN',
];

export default function MeetingPage() {
  const subtitleHistory = useMeetingStore((s) => s.subtitleHistory);
  const setPrediction = useMeetingStore((s) => s.setPrediction);
  const videoContainerRef = useRef<HTMLDivElement>(null);

  const { isLoading, isRunning, error, currentPrediction, handDetected, startCamera, stopCamera, videoRef } =
    useSignInference({
      onPrediction: setPrediction,
    });

  // Attach video element to container when running
  useEffect(() => {
    if (isRunning && videoRef.current && videoContainerRef.current) {
      const container = videoContainerRef.current;
      // Clear any existing video
      container.innerHTML = '';
      // Clone and append the video
      const video = videoRef.current.cloneNode(true) as HTMLVideoElement;
      video.style.width = '100%';
      video.style.height = '100%';
      video.style.objectFit = 'cover';
      video.style.transform = 'scaleX(-1)';
      container.appendChild(video);
    }
  }, [isRunning, videoRef]);

  const handleToggle = () => {
    if (isRunning) {
      stopCamera();
    } else {
      startCamera();
    }
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return (
    <main className="h-screen bg-gradient-to-br from-blue-950 to-blue-900 text-white flex flex-col">
      {/* Header */}
      <div className="p-4 text-xl font-semibold border-b border-blue-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-bold text-sm">
            SS
          </div>
          <span>Healthcare Sign Meeting</span>
        </div>
        <div className="flex items-center gap-4">
          {isLoading && (
            <span className="text-sm text-blue-300 animate-pulse">Loading AI models...</span>
          )}
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                isRunning ? 'bg-green-500' : 'bg-gray-500'
              }`}
            />
            <span className="text-sm text-gray-300">
              {isRunning ? 'Live' : 'Offline'}
            </span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex flex-1 gap-4 p-4 overflow-hidden">
        {/* User Video */}
        <div ref={videoContainerRef} className="flex-1 bg-black rounded-2xl overflow-hidden relative">
          {!isRunning && !isLoading && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <Video className="w-16 h-16 mx-auto mb-3 opacity-50" />
                <p className="text-lg">Camera Off</p>
              </div>
            </div>
          )}

          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <div className="text-center">
                <div className="w-10 h-10 border-3 border-blue-400 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                <p className="text-blue-300">Downloading AI models...</p>
                <p className="text-gray-400 text-xs mt-1">This may take a moment</p>
              </div>
            </div>
          )}

          {/* Status badges */}
          <div className="absolute bottom-3 left-3 flex gap-2">
            {isRunning && (
              <>
                <div
                  className={`px-3 py-1 rounded-full text-sm font-medium flex items-center gap-1.5 ${
                    handDetected
                      ? 'bg-green-600/80 text-green-100'
                      : 'bg-gray-700/80 text-gray-300'
                  }`}
                >
                  <HandMetal className="w-4 h-4" />
                  {handDetected ? 'Hand detected' : 'No hand'}
                </div>
              </>
            )}
          </div>

          {/* Current prediction overlay */}
          {currentPrediction && isRunning && (
            <div className="absolute bottom-3 right-3">
              <div className="px-4 py-2 rounded-xl text-lg font-bold bg-blue-600/90 text-white shadow-lg">
                {currentPrediction.phrase}
              </div>
            </div>
          )}
        </div>

        {/* Participant Panel */}
        <div className="w-80 bg-blue-900/50 rounded-2xl flex flex-col overflow-hidden">
          <div className="p-4 border-b border-blue-800">
            <h2 className="text-lg font-semibold">Participants</h2>
          </div>

          {/* Participant 1 - Doctor */}
          <div className="flex-1 flex flex-col items-center justify-center p-4">
            <div className="w-24 h-24 bg-gray-300 rounded-full flex items-center justify-center mb-3">
              <User className="w-12 h-12 text-gray-500" />
            </div>
            <p className="text-lg font-semibold">Dr. Sharma</p>
            <div className="flex items-center gap-1.5 mt-1">
              <div className="w-2 h-2 bg-green-500 rounded-full" />
              <span className="text-sm text-gray-300">Online</span>
            </div>
          </div>

          {/* Supported Phrases */}
          <div className="p-4 border-t border-blue-800">
            <h3 className="text-sm font-semibold text-gray-300 mb-2">Supported Phrases</h3>
            <div className="grid grid-cols-2 gap-1.5">
              {PHRASES.map((phrase) => (
                <div
                  key={phrase}
                  className={`px-2 py-1 rounded text-xs text-center transition-colors ${
                    currentPrediction?.phrase === phrase
                      ? 'bg-blue-600 text-white font-semibold'
                      : 'bg-blue-800/50 text-gray-300'
                  }`}
                >
                  {phrase}
                </div>
              ))}
            </div>
          </div>

          {/* Subtitle History */}
          <div className="h-48 border-t border-blue-800 flex flex-col">
            <div className="px-4 py-2 border-b border-blue-800 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-300">History</h3>
              <span className="text-xs text-gray-500">{subtitleHistory.length} phrases</span>
            </div>
            <div className="flex-1 overflow-y-auto p-3 space-y-2">
              {subtitleHistory.length === 0 ? (
                <p className="text-gray-500 text-xs text-center py-4">
                  No subtitles yet
                </p>
              ) : (
                [...subtitleHistory].reverse().map((entry, i) => (
                  <div key={i} className="bg-blue-800/30 rounded-lg px-3 py-2">
                    <p className="text-white font-medium text-sm">{entry.phrase}</p>
                    <p className="text-gray-400 text-xs">
                      {(entry.confidence * 100).toFixed(0)}% &middot; {entry.time.toLocaleTimeString()}
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Subtitle Bar */}
      <div className="bg-black/70 text-white text-2xl font-bold text-center py-4 min-h-[72px] flex items-center justify-center">
        {isRunning ? (
          currentPrediction ? (
            <span className="animate-pulse">{currentPrediction.phrase}</span>
          ) : (
            <span className="text-gray-400 text-lg">Sign to see subtitles...</span>
          )
        ) : (
          <span className="text-gray-400">Camera is off</span>
        )}
      </div>

      {/* Control Bar */}
      <div className="p-4 flex justify-center">
        <button
          onClick={handleToggle}
          disabled={isLoading}
          className={`px-8 py-3 rounded-xl font-semibold text-lg transition-all flex items-center gap-2 ${
            isRunning
              ? 'bg-red-600 hover:bg-red-700 text-white'
              : 'bg-green-600 hover:bg-green-700 text-white'
          } disabled:opacity-50 disabled:cursor-not-allowed shadow-lg`}
        >
          {isRunning ? (
            <>
              <VideoOff className="w-5 h-5" />
              End Meeting
            </>
          ) : (
            <>
              <Video className="w-5 h-5" />
              Start Sign Meeting
            </>
          )}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="fixed bottom-20 left-1/2 -translate-x-1/2 bg-red-600/90 text-white px-6 py-3 rounded-xl shadow-lg">
          <p className="text-sm">Error: {error}</p>
        </div>
      )}
    </main>
  );
}
