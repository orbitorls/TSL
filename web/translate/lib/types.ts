export type TrackKey = "fingerspelling" | "tsl51";

export interface Track {
  key: TrackKey;
  title: string;
}

export interface Artifact {
  name: string;
  model: string;
  labels: string;
  scaler: string;
  manifest: string | null;
}

export interface TopKEntry {
  label: string;
  p: number;
}

export interface PredictionMessage {
  type: "prediction";
  label: string;
  confidence: number;
  topk: TopKEntry[];
  topk_text: string;
  status: string;
  buffering: string | null;
  fps: number;
  transcript: string;
  confidence_hist: number[];
}

export interface TranscriptMessage {
  type: "transcript";
  text: string;
  last_token: string;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type WsMessage = PredictionMessage | TranscriptMessage | ErrorMessage | { type: "pong" };

export interface InferenceSettings {
  threshold: number;
  alpha: number;
  top_k: number;
  motion_min: number;
}
