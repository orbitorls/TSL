export type TrackKey = "fingerspelling" | "fingerspelling_dynamic" | "tsl51";

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

export interface HolisticLandmarksMessage {
  pose: number[][] | null;
  face: number[][] | null;
  left_hand: number[][] | null;
  right_hand: number[][] | null;
}

export interface HandsDetected {
  left: boolean;
  right: boolean;
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
  landmarks?: HolisticLandmarksMessage;
  hands_detected?: HandsDetected;
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
  prediction_stable_frames?: number;
  transcript_stable_frames?: number;
  transcript_debounce_s?: number;
  send_landmarks?: boolean;
}

export type Tsl51Preset = "balanced" | "accurate" | "fast";

type Tsl51PresetFields = Required<
  Pick<
    InferenceSettings,
    | "threshold"
    | "min_sign_frames"
    | "sign_end_frames"
    | "min_confidence_margin"
    | "commit_on_preview"
    | "transcript_stable_frames"
    | "transcript_debounce_s"
  >
>;

export type FsPreset = "balanced" | "strict" | "fast";

type FsPresetFields = Required<
  Pick<
    InferenceSettings,
    | "threshold"
    | "min_confidence_margin"
    | "prediction_stable_frames"
    | "alpha"
    | "transcript_stable_frames"
    | "transcript_debounce_s"
  >
>;

export const FS_PRESETS: Record<FsPreset, FsPresetFields> = {
  balanced: {
    threshold: 0.7,
    min_confidence_margin: 0.1,
    prediction_stable_frames: 2,
    alpha: 0.4,
    transcript_stable_frames: 2,
    transcript_debounce_s: 0.2,
  },
  strict: {
    threshold: 0.75,
    min_confidence_margin: 0.12,
    prediction_stable_frames: 3,
    alpha: 0.35,
    transcript_stable_frames: 2,
    transcript_debounce_s: 0.3,
  },
  fast: {
    threshold: 0.65,
    min_confidence_margin: 0.08,
    prediction_stable_frames: 1,
    alpha: 0.45,
    transcript_stable_frames: 1,
    transcript_debounce_s: 0.15,
  },
};

export const TSL51_PRESETS: Record<Tsl51Preset, Tsl51PresetFields> = {
  balanced: {
    threshold: 0.62,
    min_sign_frames: 4,
    sign_end_frames: 3,
    min_confidence_margin: 0.1,
    commit_on_preview: true,
    transcript_stable_frames: 1,
    transcript_debounce_s: 0.2,
  },
  accurate: {
    threshold: 0.65,
    min_sign_frames: 6,
    sign_end_frames: 5,
    min_confidence_margin: 0.12,
    commit_on_preview: false,
    transcript_stable_frames: 2,
    transcript_debounce_s: 0.4,
  },
  fast: {
    threshold: 0.55,
    min_sign_frames: 3,
    sign_end_frames: 5,
    min_confidence_margin: 0.08,
    commit_on_preview: true,
    transcript_stable_frames: 1,
    transcript_debounce_s: 0.2,
  },
};

export type FsDynamicPreset = "balanced" | "accurate" | "fast";

type FsDynamicPresetFields = Required<
  Pick<
    InferenceSettings,
    | "threshold"
    | "min_sign_frames"
    | "sign_end_frames"
    | "min_confidence_margin"
    | "commit_on_preview"
    | "transcript_stable_frames"
    | "transcript_debounce_s"
  >
>;

export const FS_DYNAMIC_PRESETS: Record<FsDynamicPreset, FsDynamicPresetFields> = {
  balanced: {
    threshold: 0.65,
    min_sign_frames: 12,
    sign_end_frames: 5,
    min_confidence_margin: 0.10,
    commit_on_preview: true,
    transcript_stable_frames: 1,
    transcript_debounce_s: 0.2,
  },
  accurate: {
    threshold: 0.7,
    min_sign_frames: 16,
    sign_end_frames: 6,
    min_confidence_margin: 0.12,
    commit_on_preview: false,
    transcript_stable_frames: 2,
    transcript_debounce_s: 0.35,
  },
  fast: {
    threshold: 0.55,
    min_sign_frames: 8,
    sign_end_frames: 4,
    min_confidence_margin: 0.08,
    commit_on_preview: true,
    transcript_stable_frames: 1,
    transcript_debounce_s: 0.15,
  },
};
