"use client";

import type { Artifact, InferenceSettings, Track, TrackKey, Tsl51Preset } from "@/lib/types";

interface Props {
  tracks: Track[];
  track: TrackKey;
  onTrackChange: (k: TrackKey) => void;
  artifacts: Artifact[];
  selectedArtifact: string;
  onArtifactChange: (name: string) => void;
  settings: InferenceSettings;
  onSettingsChange: (s: Partial<InferenceSettings>) => void;
  tsl51Preset: Tsl51Preset;
  onTsl51PresetChange: (preset: Tsl51Preset) => void;
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
  tsl51Preset,
  onTsl51PresetChange,
  modelLoaded,
  modelInfo,
  onLoad,
  loading,
}: Props) {
  const selectedTrack = tracks.find((t) => t.key === track);
  const canLoad = Boolean(selectedArtifact) && !loading;

  return (
    <aside className="space-y-4 rounded-3xl border border-border bg-panel/95 p-4 shadow-sm ring-1 ring-white/60 lg:sticky lg:top-4">
      <div className="rounded-2xl bg-brand px-4 py-3 text-white shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/70">Setup</p>
        <h2 className="mt-1 text-xl font-bold">ควบคุมระบบ</h2>
        <p className="mt-1 text-sm leading-6 text-white/80">เลือกแทร็ก โมเดล และปรับค่าการทำนายก่อนเริ่มกล้อง</p>
      </div>

      <section className="space-y-3 rounded-2xl border border-border bg-page/50 p-3">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-bold text-text">1. เลือกงานแปล</h3>
          <span className="rounded-full bg-white px-2.5 py-1 text-[11px] font-semibold text-subtle shadow-sm">
            {selectedTrack?.title ?? "กำลังโหลด"}
          </span>
        </div>

        <label className="block text-sm">
          <span className="text-subtle">แทร็กโมเดล</span>
          <select
            className="mt-1 w-full rounded-xl border border-border bg-white px-3 py-2 font-medium text-text outline-none transition focus:border-brand focus:ring-2 focus:ring-brand/15 disabled:cursor-not-allowed disabled:bg-zinc-100 disabled:text-subtle"
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
            className="mt-1 w-full rounded-xl border border-border bg-white px-3 py-2 text-xs font-medium text-text outline-none transition focus:border-brand focus:ring-2 focus:ring-brand/15 disabled:cursor-not-allowed disabled:bg-zinc-100 disabled:text-subtle"
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
      </section>

      <section className="space-y-3 rounded-2xl border border-border bg-white p-3">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-bold text-text">2. โหลดโมเดล</h3>
          <span
            className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${
              modelLoaded
                ? "bg-emerald-50 text-emerald-700"
                : loading
                  ? "bg-amber-50 text-amber-700"
                  : "bg-brand-ghost text-subtle"
            }`}
          >
            {modelLoaded ? "พร้อมใช้" : loading ? "กำลังโหลด" : "ยังไม่โหลด"}
          </span>
        </div>

        <button
          type="button"
          onClick={onLoad}
          disabled={!canLoad}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-brand py-3 font-semibold text-white shadow-sm transition hover:bg-brand/95 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading && <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-white" aria-hidden />}
          {loading ? "กำลังโหลดโมเดล..." : modelLoaded ? "โหลดโมเดลอีกครั้ง" : "โหลดโมเดล"}
        </button>

        {modelLoaded && modelInfo ? (
          <p className="rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs leading-5 text-emerald-800">
            {modelInfo}
          </p>
        ) : (
          <p className="rounded-xl border border-dashed border-border bg-page/60 px-3 py-2 text-xs leading-5 text-subtle">
            ต้องโหลดโมเดลก่อนเปิดกล้อง เพื่อป้องกันการทำนายผิดแทร็ก
          </p>
        )}
      </section>

      <section className="space-y-3 rounded-2xl border border-border bg-page/50 p-3 text-sm">
        <div>
          <h3 className="text-sm font-bold text-text">3. ค่าการทำนาย</h3>
          <p className="mt-0.5 text-xs text-subtle">ปรับอย่างค่อยเป็นค่อยไปเพื่อคงความนิ่งของผลลัพธ์</p>
        </div>

        <label className="block">
          <span className="flex items-center justify-between gap-3 text-subtle">
            <span>เกณฑ์ความมั่นใจ</span>
            <strong className="text-text">{settings.threshold.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min={0.1}
            max={0.99}
            step={0.01}
            value={settings.threshold}
            onChange={(e) => onSettingsChange({ threshold: Number(e.target.value) })}
            className="mt-1 w-full accent-brand"
          />
        </label>
        <label className="block">
          <span className="flex items-center justify-between gap-3 text-subtle">
            <span>Smoothing alpha</span>
            <strong className="text-text">{settings.alpha.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min={0.05}
            max={1}
            step={0.05}
            value={settings.alpha}
            onChange={(e) => onSettingsChange({ alpha: Number(e.target.value) })}
            className="mt-1 w-full accent-brand"
          />
        </label>
        <label className="block">
          <span className="flex items-center justify-between gap-3 text-subtle">
            <span>Top-K</span>
            <strong className="text-text">{settings.top_k}</strong>
          </span>
          <input
            type="range"
            min={1}
            max={5}
            step={1}
            value={settings.top_k}
            onChange={(e) => onSettingsChange({ top_k: Number(e.target.value) })}
            className="mt-1 w-full accent-brand"
          />
        </label>
        <label className="block">
          <span className="flex items-center justify-between gap-3 text-subtle">
            <span>ความเคลื่อนไหวขั้นต่ำ</span>
            <strong className="text-text">{settings.motion_min.toFixed(3)}</strong>
          </span>
          <input
            type="range"
            min={0.001}
            max={0.05}
            step={0.001}
            value={settings.motion_min}
            onChange={(e) => onSettingsChange({ motion_min: Number(e.target.value) })}
            className="mt-1 w-full accent-brand"
          />
        </label>
        {track === "tsl51" && (
          <>
            <div className="rounded-xl border border-border bg-white p-2">
              <p className="mb-2 text-xs font-semibold text-subtle">โหมด TSL-51</p>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => onTsl51PresetChange("accurate")}
                  className={`rounded-lg px-3 py-2 text-xs font-semibold transition ${
                    tsl51Preset === "accurate"
                      ? "bg-brand text-white shadow-sm"
                      : "border border-border bg-page text-subtle hover:bg-white"
                  }`}
                >
                  แม่นยำ
                </button>
                <button
                  type="button"
                  onClick={() => onTsl51PresetChange("fast")}
                  className={`rounded-lg px-3 py-2 text-xs font-semibold transition ${
                    tsl51Preset === "fast"
                      ? "bg-brand text-white shadow-sm"
                      : "border border-border bg-page text-subtle hover:bg-white"
                  }`}
                >
                  เร็ว
                </button>
              </div>
              <p className="mt-2 text-xs leading-5 text-subtle">
                {tsl51Preset === "accurate"
                  ? "รอจบท่าก่อน commit · แนะนำสำหรับใช้งานจริง"
                  : "preview เร็วขึ้น · อาจทายผิดบ่อยกว่า"}
              </p>
            </div>
            <label className="block">
              <span className="flex items-center justify-between gap-3 text-subtle">
                <span>เฟรมขั้นต่ำก่อนเริ่มทำนาย</span>
                <strong className="text-text">{settings.min_sign_frames ?? 6}</strong>
              </span>
              <input
                type="range"
                min={1}
                max={15}
                step={1}
                value={settings.min_sign_frames ?? 6}
                onChange={(e) => onSettingsChange({ min_sign_frames: Number(e.target.value) })}
                className="mt-1 w-full accent-brand"
              />
            </label>
            <label className="block">
              <span className="flex items-center justify-between gap-3 text-subtle">
                <span>เฟรมนิ่งก่อนปิดคำ</span>
                <strong className="text-text">{settings.sign_end_frames ?? 5}</strong>
              </span>
              <input
                type="range"
                min={3}
                max={12}
                step={1}
                value={settings.sign_end_frames ?? 5}
                onChange={(e) => onSettingsChange({ sign_end_frames: Number(e.target.value) })}
                className="mt-1 w-full accent-brand"
              />
            </label>
            <label className="block">
              <span className="flex items-center justify-between gap-3 text-subtle">
                <span>ช่องว่าง Top-1 / Top-2 ขั้นต่ำ</span>
                <strong className="text-text">{(settings.min_confidence_margin ?? 0.12).toFixed(2)}</strong>
              </span>
              <input
                type="range"
                min={0.05}
                max={0.3}
                step={0.01}
                value={settings.min_confidence_margin ?? 0.12}
                onChange={(e) => onSettingsChange({ min_confidence_margin: Number(e.target.value) })}
                className="mt-1 w-full accent-brand"
              />
            </label>
          </>
        )}
      </section>
    </aside>
  );
}
