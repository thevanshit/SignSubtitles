import { create } from 'zustand';

export interface SignPrediction {
  phrase: string;
  confidence: number;
}

interface MeetingState {
  isMeetingActive: boolean;
  isLoading: boolean;
  error: string | null;
  currentPrediction: SignPrediction | null;
  subtitleHistory: Array<{ phrase: string; time: Date; confidence: number }>;
  handDetected: boolean;
  setMeetingActive: (active: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setPrediction: (prediction: SignPrediction | null) => void;
  setHandDetected: (detected: boolean) => void;
}

export const useMeetingStore = create<MeetingState>((set) => ({
  isMeetingActive: false,
  isLoading: true,
  error: null,
  currentPrediction: null,
  subtitleHistory: [],
  handDetected: false,
  setMeetingActive: (active) => set({ isMeetingActive: active }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  setPrediction: (prediction) =>
    set((state) => {
      if (prediction) {
        const newEntry = { ...prediction, time: new Date() };
        const history = [...state.subtitleHistory, newEntry].slice(-50);
        return { currentPrediction: prediction, subtitleHistory: history };
      }
      return { currentPrediction: null };
    }),
  setHandDetected: (detected) => set({ handDetected: detected }),
}));
