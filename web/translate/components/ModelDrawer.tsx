"use client";

import type { Artifact, FsDynamicPreset, FsPreset, InferenceSettings, Track, TrackKey, Tsl51Preset } from "@/lib/types";
import { STATUS } from "@/lib/status";

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
  fsPreset: FsPreset;
  onFsPresetChange: (preset: FsPreset) => void;
  fsDynamicPreset: FsDynamicPreset;
  onFsDynamicPresetChange: (preset: FsDynamicPreset) => void;
  modelLoaded: boolean;
  modelInfo: string | null;
  onLoad: () => void;
  loading: boolean;
  showSkeleton: boolean;
  onShowSkeletonChange: (value: boolean) => void;
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
  fsPreset,
  onFsPresetChange,
  fsDynamicPreset,
  onFsDynamicPresetChange,
  modelLoaded,
  modelInfo,
  onLoad,
  loading,
  showSkeleton,
  onShowSkeletonChange,
}: Props) {
  const selectedTrack = tracks.find((t) => t.key === track);
  const canLoad = Boolean(selectedArtifact) && !loading;

  // Preset button helper
  const presetBtn = (active: boolean) =>
    `rounded-field px-2 py-2 text-xs font-semibold transition-colors ${
      active
        ? "bg-brand text-brand-fg shadow-card"
        : "border border-line bg-panel-2 text-subtle hover:bg-panel hover:text-text"
    }`;

  return (
    <aside className="space-y-3 rounded-panel border border-line bg-panel p-4 shadow-card lg:sticky lg:top-4">
      {/* Brand header */}
      <div className="rounded-card bg-brand px-4 py-3.5">
        <h2 className="text-lg font-bold text-brand-fg">ควบคุมระบบ</h2>
        <p className="mt-0.5 text-xs leading-5 text-brand-fg/75">
          เลือกแทร็ก โมเดล และปรับค่าก่อนเริ่มกล้อง
        </p>
      </div>

      {/* ── Section 1: Track + Artifact selection ── */}
      <section className="space-y-3 rounded-card border border-line bg-panel-2 p-3">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-ink">1. เลือกงานแปล</h3>
          <span className="rounded-full border border-line bg-panel px-2.5 py-0.5 text-[11px] font-semibold text-subtle">
            {selectedTrack?.title ?? "กำลังโหลด"}
          </span>
        </div>

        <label className="block text-sm">
          <span className="text-subtle">แทร็กโมเดล</span>
          <select
            className="mt-1 w-full rounded-field border border-border bg-panel px-3 py-2 text-sm font-medium text-text outline-none transition focus:border-brand focus:ring-2 focus:ring-brand/20 disabled:cursor-not-allowed disabled:bg-panel-2 disabled:text-muted"
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
            className="mt-1 w-full rounded-field border border-border bg-panel px-3 py-2 text-xs font-medium text-text outline-none transition focus:border-brand focus:ring-2 focus:ring-brand/20 disabled:cursor-not-allowed disabled:bg-panel-2 disabled:text-muted"
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

      {/* ── Section 2: Load model ── */}
      <section className="space-y-3 rounded-card border border-line bg-panel p-3">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-ink">2. โหลดโมเดล</h3>
          <span
            className={`rounded-full border px-2.5 py-0.5 text-[11px] font-semibold ${
              modelLoaded
                ? `${STATUS.success.bg} ${STATUS.success.border} ${STATUS.success.text}`
                : loading
                  ? `${STATUS.connecting.bg} ${STATUS.connecting.border} ${STATUS.connecting.text}`
                  : `${STATUS.idle.bg} ${STATUS.idle.border} ${STATUS.idle.text}`
            }`}
          >
            {modelLoaded ? "พร้อมใช้" : loading ? "กำลังโหลด" : "ยังไม่โหลด"}
          </span>
        </div>

        <button
          type="button"
          onClick={onLoad}
          disabled={!canLoad}
          className="flex w-full items-center justify-center gap-2 rounded-field bg-brand py-2.5 text-sm font-semibold text-brand-fg shadow-card hover:bg-brand-strong disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading && (
            <span className="h-2 w-2 animate-pulse rounded-full bg-brand-fg" aria-hidden />
          )}
          {loading ? "กำลังโหลดโมเดล..." : modelLoaded ? "โหลดโมเดลอีกครั้ง" : "โหลดโมเดล"}
        </button>

        {modelLoaded && modelInfo ? (
          <p className={`rounded-field border px-3 py-2 text-xs leading-5 ${STATUS.success.border} ${STATUS.success.bg} ${STATUS.success.text}`}>
            {modelInfo}
          </p>
        ) : (
          <p className="rounded-field border border-line bg-panel-2 px-3 py-2 text-xs leading-5 text-subtle">
            ต้องโหลดโมเดลก่อนเปิดกล้อง เพื่อป้องกันการทำนายผิดแทร็ก
          </p>
        )}
      </section>

      {/* ── Section 3: Inference settings ── */}
      <section className="space-y-3 rounded-card border border-line bg-panel-2 p-3 text-sm">
        <div>
          <h3 className="text-sm font-semibold text-ink">3. ค่าการทำนาย</h3>
          <p className="mt-0.5 text-xs text-subtle">ปรับอย่างค่อยเป็นค่อยไปเพื่อคงความนิ่ง</p>
        </div>

        {/* Core sliders */}
        <label className="block">
          <span className="flex items-center justify-between gap-3 text-subtle">
            <span>เกณฑ์ความมั่นใจ</span>
            <strong className="font-mono font-medium text-text">{settings.threshold.toFixed(2)}</strong>
          </span>
          <input
            type="range" min={0.1} max={0.99} step={0.01}
            value={settings.threshold}
            onChange={(e) => onSettingsChange({ threshold: Number(e.target.value) })}
            className="mt-1 w-full accent-brand"
          />
        </label>
        <label className="block">
          <span className="flex items-center justify-between gap-3 text-subtle">
            <span>Smoothing alpha</span>
            <strong className="font-mono font-medium text-text">{settings.alpha.toFixed(2)}</strong>
          </span>
          <input
            type="range" min={0.05} max={1} step={0.05}
            value={settings.alpha}
            onChange={(e) => onSettingsChange({ alpha: Number(e.target.value) })}
            className="mt-1 w-full accent-brand"
          />
        </label>
        <label className="block">
          <span className="flex items-center justify-between gap-3 text-subtle">
            <span>Top-K</span>
            <strong className="font-mono font-medium text-text">{settings.top_k}</strong>
          </span>
          <input
            type="range" min={1} max={5} step={1}
            value={settings.top_k}
            onChange={(e) => onSettingsChange({ top_k: Number(e.target.value) })}
            className="mt-1 w-full accent-brand"
          />
        </label>
        <label className="block">
          <span className="flex items-center justify-between gap-3 text-subtle">
            <span>ความเคลื่อนไหวขั้นต่ำ</span>
            <strong className="font-mono font-medium text-text">{settings.motion_min.toFixed(3)}</strong>
          </span>
          <input
            type="range" min={0.001} max={0.05} step={0.001}
            value={settings.motion_min}
            onChange={(e) => onSettingsChange({ motion_min: Number(e.target.value) })}
            className="mt-1 w-full accent-brand"
          />
        </label>

        {/* Fingerspelling presets */}
        {track === "fingerspelling" && (
          <>
            <div className="rounded-card border border-line bg-panel p-2.5">
              <p className="mb-2 text-xs font-semibold text-subtle">โหมดสะกดนิ้ว</p>
              <div className="grid grid-cols-3 gap-2">
                {(["balanced", "strict", "fast"] as const).map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => onFsPresetChange(p)}
                    className={presetBtn(fsPreset === p)}
                  >
                    {p === "balanced" ? "สมดุล" : p === "strict" ? "ชัดเจน" : "เร็ว"}
                  </button>
                ))}
              </div>
              <p className="mt-2 text-xs leading-5 text-subtle">
                {fsPreset === "balanced"
                  ? "แนะนำ · margin + คงท่า 2 เฟรม ลดสับ ก/บ"
                  : fsPreset === "strict"
                    ? "ท่าไม่ชัดจะไม่เดา · ต้องถือนิ่ง 3 เฟรม"
                    : "ตอบเร็ว · อาจสับตัวใกล้กันมากขึ้น"}
              </p>
            </div>
            <label className="block">
              <span className="flex items-center justify-between gap-3 text-subtle">
                <span>ช่องว่าง Top-1/Top-2 ขั้นต่ำ</span>
                <strong className="font-mono font-medium text-text">{(settings.min_confidence_margin ?? 0.1).toFixed(2)}</strong>
              </span>
              <input
                type="range" min={0.05} max={0.25} step={0.01}
                value={settings.min_confidence_margin ?? 0.1}
                onChange={(e) => onSettingsChange({ min_confidence_margin: Number(e.target.value) })}
                className="mt-1 w-full accent-brand"
              />
            </label>
            <label className="block">
              <span className="flex items-center justify-between gap-3 text-subtle">
                <span>เฟรมคงท่าก่อน commit</span>
                <strong className="font-mono font-medium text-text">{settings.prediction_stable_frames ?? 2}</strong>
              </span>
              <input
                type="range" min={1} max={5} step={1}
                value={settings.prediction_stable_frames ?? 2}
                onChange={(e) => onSettingsChange({ prediction_stable_frames: Number(e.target.value) })}
                className="mt-1 w-full accent-brand"
              />
            </label>
          </>
        )}

        {/* Fingerspelling dynamic presets */}
        {track === "fingerspelling_dynamic" && (
          <>
            <div className="rounded-card border border-line bg-panel p-2.5">
              <p className="mb-2 text-xs font-semibold text-subtle">โหมดสะกดนิ้วจังหวะ</p>
              <div className="grid grid-cols-3 gap-2">
                {(["balanced", "accurate", "fast"] as const).map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => onFsDynamicPresetChange(p)}
                    className={presetBtn(fsDynamicPreset === p)}
                  >
                    {p === "balanced" ? "สมดุล" : p === "accurate" ? "แม่นยำ" : "เร็ว"}
                  </button>
                ))}
              </div>
              <p className="mt-2 text-xs leading-5 text-subtle">
                {fsDynamicPreset === "balanced"
                  ? "แนะนำ · ใช้กับท่าสองจังหวะ เช่น ก.ไฟ ข.ไฟ"
                  : fsDynamicPreset === "accurate"
                    ? "รอจบท่าก่อน commit · แม่นยำกว่าเมื่อเซ็นช้า"
                    : "preview เร็ว · อาจทายผิดท่าที่ใกล้กัน"}
              </p>
            </div>
            <label className="block">
              <span className="flex items-center justify-between gap-3 text-subtle">
                <span>เฟรมขั้นต่ำก่อนเริ่มทำนาย</span>
                <strong className="font-mono font-medium text-text">{settings.min_sign_frames ?? 12}</strong>
              </span>
              <input
                type="range" min={1} max={15} step={1}
                value={settings.min_sign_frames ?? 12}
                onChange={(e) => onSettingsChange({ min_sign_frames: Number(e.target.value) })}
                className="mt-1 w-full accent-brand"
              />
            </label>
            <label className="block">
              <span className="flex items-center justify-between gap-3 text-subtle">
                <span>เฟรมนิ่งก่อนปิดคำ</span>
                <strong className="font-mono font-medium text-text">{settings.sign_end_frames ?? 5}</strong>
              </span>
              <input
                type="range" min={3} max={12} step={1}
                value={settings.sign_end_frames ?? 5}
                onChange={(e) => onSettingsChange({ sign_end_frames: Number(e.target.value) })}
                className="mt-1 w-full accent-brand"
              />
            </label>
            <label className="block">
              <span className="flex items-center justify-between gap-3 text-subtle">
                <span>ช่องว่าง Top-1/Top-2 ขั้นต่ำ</span>
                <strong className="font-mono font-medium text-text">{(settings.min_confidence_margin ?? 0.1).toFixed(2)}</strong>
              </span>
              <input
                type="range" min={0.05} max={0.3} step={0.01}
                value={settings.min_confidence_margin ?? 0.1}
                onChange={(e) => onSettingsChange({ min_confidence_margin: Number(e.target.value) })}
                className="mt-1 w-full accent-brand"
              />
            </label>
          </>
        )}

        {/* TSL-51 presets */}
        {track === "tsl51" && (
          <>
            <div className="rounded-card border border-line bg-panel p-2.5">
              <p className="mb-2 text-xs font-semibold text-subtle">โหมด TSL-51</p>
              <div className="grid grid-cols-3 gap-2">
                {(["balanced", "accurate", "fast"] as const).map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => onTsl51PresetChange(p)}
                    className={presetBtn(tsl51Preset === p)}
                  >
                    {p === "balanced" ? "สมดุล" : p === "accurate" ? "แม่นยำ" : "เร็ว"}
                  </button>
                ))}
              </div>
              <p className="mt-2 text-xs leading-5 text-subtle">
                {tsl51Preset === "balanced"
                  ? "แนะนำ · เร็วสำหรับหลายคำ ยังใช้ margin gate · หยุดนิ่งสั้น ~0.3 วิ"
                  : tsl51Preset === "accurate"
                    ? "รอจบท่าก่อน commit · ท่าเดี่ยวช้าแต่มั่นใจกว่า"
                    : "preview เร็วที่สุด · อาจทายผิดบ่อยกว่า"}
              </p>
            </div>
            <label className="block">
              <span className="flex items-center justify-between gap-3 text-subtle">
                <span>เฟรมขั้นต่ำก่อนเริ่มทำนาย</span>
                <strong className="font-mono font-medium text-text">{settings.min_sign_frames ?? 6}</strong>
              </span>
              <input
                type="range" min={1} max={15} step={1}
                value={settings.min_sign_frames ?? 6}
                onChange={(e) => onSettingsChange({ min_sign_frames: Number(e.target.value) })}
                className="mt-1 w-full accent-brand"
              />
            </label>
            <label className="block">
              <span className="flex items-center justify-between gap-3 text-subtle">
                <span>เฟรมนิ่งก่อนปิดคำ</span>
                <strong className="font-mono font-medium text-text">{settings.sign_end_frames ?? 5}</strong>
              </span>
              <input
                type="range" min={3} max={12} step={1}
                value={settings.sign_end_frames ?? 5}
                onChange={(e) => onSettingsChange({ sign_end_frames: Number(e.target.value) })}
                className="mt-1 w-full accent-brand"
              />
            </label>
            <label className="block">
              <span className="flex items-center justify-between gap-3 text-subtle">
                <span>ช่องว่าง Top-1/Top-2 ขั้นต่ำ</span>
                <strong className="font-mono font-medium text-text">{(settings.min_confidence_margin ?? 0.12).toFixed(2)}</strong>
              </span>
              <input
                type="range" min={0.05} max={0.3} step={0.01}
                value={settings.min_confidence_margin ?? 0.12}
                onChange={(e) => onSettingsChange({ min_confidence_margin: Number(e.target.value) })}
                className="mt-1 w-full accent-brand"
              />
            </label>
          </>
        )}

        {/* Skeleton toggle */}
        <label className="flex cursor-pointer items-center justify-between gap-3 rounded-card border border-line bg-panel px-3 py-2.5">
          <span>
            <span className="block font-medium text-text">แสดง skeleton</span>
            <span className="text-xs text-subtle">โครงร่างจาก server (ตรงกับที่โมเดลเห็น)</span>
          </span>
          <input
            type="checkbox"
            checked={showSkeleton}
            onChange={(e) => onShowSkeletonChange(e.target.checked)}
            className="h-4 w-4 accent-brand"
          />
        </label>

        <p className="rounded-card border border-line bg-panel px-3 py-2 text-xs leading-5 text-subtle">
          ถ้า skeleton ไม่ขึ้นแต่ทายได้: ตรวจแสงและให้มืออยู่ในเฟรม · หยุดนิ่งสั้นหลังจบท่า
        </p>
      </section>
    </aside>
  );
}
