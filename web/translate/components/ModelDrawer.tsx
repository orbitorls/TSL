"use client";

import type { Artifact, InferenceSettings, Track, TrackKey } from "@/lib/types";

interface Props {
  tracks: Track[];
  track: TrackKey;
  onTrackChange: (k: TrackKey) => void;
  artifacts: Artifact[];
  selectedArtifact: string;
  onArtifactChange: (name: string) => void;
  settings: InferenceSettings;
  onSettingsChange: (s: Partial<InferenceSettings>) => void;
  modelLoaded: boolean;
  modelInfo: string | null;
  onLoad: () => void;
  loading: boolean;
}

export function ModelDrawer({
  tracks,
  track,
  onTrackChange,
  artifacts,
  selectedArtifact,
  onArtifactChange,
  settings,
  onSettingsChange,
  modelLoaded,
  modelInfo,
  onLoad,
  loading,
}: Props) {
  return (
    <aside className="space-y-4 rounded-2xl border border-border bg-panel p-4 shadow-sm lg:sticky lg:top-4">
      <h2 className="font-bold text-brand">ควบคุมระบบ</h2>

      <label className="block text-sm">
        <span className="text-subtle">แทร็กโมเดล</span>
        <select
          className="mt-1 w-full rounded-xl border border-border bg-white px-3 py-2"
          value={track}
          onChange={(e) => onTrackChange(e.target.value as TrackKey)}
          disabled={loading}
        >
          {tracks.map((t) => (
            <option key={t.key} value={t.key}>
              {t.title}
            </option>
          ))}
        </select>
      </label>

      <label className="block text-sm">
        <span className="text-subtle">ชุดไฟล์ที่ค้นพบ</span>
        <select
          className="mt-1 w-full rounded-xl border border-border bg-white px-3 py-2 text-xs"
          value={selectedArtifact}
          onChange={(e) => onArtifactChange(e.target.value)}
          disabled={!artifacts.length || loading}
        >
          {artifacts.length === 0 ? (
            <option value="">ไม่พบ artifacts</option>
          ) : (
            artifacts.map((a) => (
              <option key={a.name} value={a.name}>
                {a.name}
              </option>
            ))
          )}
        </select>
      </label>

      <button
        type="button"
        onClick={onLoad}
        disabled={loading || !selectedArtifact}
        className="w-full rounded-xl bg-brand py-3 font-semibold text-white disabled:opacity-50"
      >
        {loading ? "กำลังโหลด..." : "โหลดโมเดล"}
      </button>

      {modelLoaded && modelInfo && (
        <p className="rounded-lg bg-emerald-50 px-3 py-2 text-xs text-emerald-800">{modelInfo}</p>
      )}

      <div className="space-y-3 border-t border-border pt-3 text-sm">
        <label className="block">
          <span className="text-subtle">เกณฑ์ความมั่นใจ ({settings.threshold.toFixed(2)})</span>
          <input
            type="range"
            min={0.1}
            max={0.99}
            step={0.01}
            value={settings.threshold}
            onChange={(e) => onSettingsChange({ threshold: Number(e.target.value) })}
            className="mt-1 w-full"
          />
        </label>
        <label className="block">
          <span className="text-subtle">Smoothing alpha ({settings.alpha.toFixed(2)})</span>
          <input
            type="range"
            min={0.05}
            max={1}
            step={0.05}
            value={settings.alpha}
            onChange={(e) => onSettingsChange({ alpha: Number(e.target.value) })}
            className="mt-1 w-full"
          />
        </label>
        <label className="block">
          <span className="text-subtle">Top-K ({settings.top_k})</span>
          <input
            type="range"
            min={1}
            max={5}
            step={1}
            value={settings.top_k}
            onChange={(e) => onSettingsChange({ top_k: Number(e.target.value) })}
            className="mt-1 w-full"
          />
        </label>
        {track === "tsl51" && (
          <label className="block">
            <span className="text-subtle">ความเคลื่อนไหวขั้นต่ำ ({settings.motion_min.toFixed(3)})</span>
            <input
              type="range"
              min={0.001}
              max={0.05}
              step={0.001}
              value={settings.motion_min}
              onChange={(e) => onSettingsChange({ motion_min: Number(e.target.value) })}
              className="mt-1 w-full"
            />
          </label>
        )}
      </div>
    </aside>
  );
}
