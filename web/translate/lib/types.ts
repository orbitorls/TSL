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
  committed_label: string | null;
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
  min_sign_frames?: number;
  sign_end_frames?: number;
  min_confidence_margin?: number;
  commit_on_preview?: boolean;
}

export type Tsl51Preset = "accurate" | "fast";

export const TSL51_PRESETS: Record<Tsl51Preset, Required<Pick<
  InferenceSettings,
  "threshold" | "min_sign_frames" | "sign_end_frames" | "min_confidence_margin" | "commit_on_preview"
>>> = {
  accurate: {
    threshold: 0.65,
    min_sign_frames: 6,
    sign_end_frames: 5,
    min_confidence_margin: 0.12,
    commit_on_preview: false,
  },
  fast: {
    threshold: 0.55,
    min_sign_frames: 3,
    sign_end_frames: 5,
    min_confidence_margin: 0.08,
    commit_on_preview: true,
  },
};
