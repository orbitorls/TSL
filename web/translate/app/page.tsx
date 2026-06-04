"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { CameraStage } from "@/components/CameraStage";
import { ConfidenceChart } from "@/components/ConfidenceChart";
import { ControlsBar } from "@/components/ControlsBar";
import { ModelDrawer } from "@/components/ModelDrawer";
import { StatusChip } from "@/components/StatusChip";
import { Stepper } from "@/components/Stepper";
import { ThemeToggle } from "@/components/ThemeToggle";
import { TranscriptPanel } from "@/components/TranscriptPanel";
import {
  createSession,
  fetchArtifacts,
  fetchTracks,
  loadModel,
  patchSettings,
  transcriptAction,
} from "@/lib/api";
import type { Artifact, FsDynamicPreset, FsPreset, InferenceSettings, Track, TrackKey, Tsl51Preset } from "@/lib/types";
import { FS_DYNAMIC_PRESETS, FS_PRESETS, TSL51_PRESETS } from "@/lib/types";
import type { Status } from "@/lib/status";
import { useInferenceWs } from "@/lib/ws";
import { glossLabel } from "@/lib/gloss";

const defaultSettings = (track: TrackKey): InferenceSettings => ({
  threshold: track === "fingerspelling"
    ? FS_PRESETS.balanced.threshold
    : track === "fingerspelling_dynamic"
      ? FS_DYNAMIC_PRESETS.balanced.threshold
      : TSL51_PRESETS.balanced.threshold,
  alpha: track === "fingerspelling" ? FS_PRESETS.balanced.alpha : 0.4,
  top_k: 3,
  motion_min: 0.008,
  send_landmarks: true,
  ...(track === "fingerspelling" ? FS_PRESETS.balanced : {}),
  ...(track === "fingerspelling_dynamic" ? FS_DYNAMIC_PRESETS.balanced : {}),
  ...(track === "tsl51" ? TSL51_PRESETS.balanced : {}),
});

export default function TranslatePage() {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [track, setTrack] = useState<TrackKey>("fingerspelling");
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [selectedArtifact, setSelectedArtifact] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [settings, setSettings] = useState<InferenceSettings>(defaultSettings("fingerspelling"));
  const [tsl51Preset, setTsl51Preset] = useState<Tsl51Preset>("balanced");
  const [fsPreset, setFsPreset] = useState<FsPreset>("balanced");
  const [fsDynamicPreset, setFsDynamicPreset] = useState<FsDynamicPreset>("balanced");
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelInfo, setModelInfo] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showDrawer, setShowDrawer] = useState(false); // mobile drawer toggle

  const { state, attachVideo, setTranscript } = useInferenceWs(sessionId, streaming);

  const step = useMemo(() => {
    if (streaming) return 4;
    if (modelLoaded) return 3;
    if (selectedArtifact) return 2;
    return 1;
  }, [streaming, modelLoaded, selectedArtifact]);

  const selectedTrackTitle = useMemo(
    () => tracks.find((t) => t.key === track)?.title ?? (track === "tsl51" ? "TSL-51" : "Fingerspelling"),
    [track, tracks],
  );

  // Derived status values for StatusChip components
  const connectionStatus: Status = state.connected && streaming ? "live" : streaming ? "connecting" : "idle";
  const modelStatus: Status = modelLoaded ? "success" : selectedArtifact ? "connecting" : "idle";
  const connectionLabel = state.connected && streaming ? "สตรีมสด" : streaming ? "กำลังเชื่อมต่อ" : "พร้อมเริ่ม";
  const modelLabel = modelLoaded ? "โมเดลพร้อม" : selectedArtifact ? "รอโหลดโมเดล" : "รอเลือกไฟล์";
  const artifactLabel = selectedArtifact || "ไม่พบ artifact";

  useEffect(() => {
    fetchTracks()
      .then(setTracks)
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    setSettings(defaultSettings(track));
    setTsl51Preset("balanced");
    setFsPreset("balanced");
    setFsDynamicPreset("balanced");
    setModelLoaded(false);
    setModelInfo(null);
    setStreaming(false);
    fetchArtifacts(track)
      .then((list) => {
        setArtifacts(list);
        setSelectedArtifact(list[0]?.name ?? "");
      })
      .catch((e) => setError(String(e)));
    createSession(track)
      .then((s) => setSessionId(s.session_id))
      .catch((e) => setError(String(e)));
  }, [track]);

  useEffect(() => {
    if (!sessionId || !modelLoaded) return;
    const t = setTimeout(() => {
      patchSettings(sessionId, settings).catch(() => {});
    }, 300);
    return () => clearTimeout(t);
  }, [sessionId, modelLoaded, settings]);

  useEffect(() => {
    if (!sessionId || !modelLoaded) return;
    patchSettings(sessionId, { send_landmarks: showSkeleton }).catch(() => {});
  }, [sessionId, modelLoaded, showSkeleton]);

  const handleLoad = async () => {
    if (!sessionId || !selectedArtifact) return;
    setLoading(true);
    setError(null);
    try {
      const res = await loadModel(sessionId, selectedArtifact);
      setModelLoaded(true);
      setModelInfo(`โหลดสำเร็จ · ${res.backend} · ${res.num_classes} คลาส · ${res.load_time_ms.toFixed(0)} ms`);
      await patchSettings(sessionId, settings);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleTranscript = async (action: "space" | "backspace" | "clear") => {
    if (!sessionId) return;
    try {
      const res = await transcriptAction(sessionId, action);
      setTranscript(res.text);
    } catch (e) {
      setError(String(e));
    }
  };

  const onSettingsChange = useCallback((partial: Partial<InferenceSettings>) => {
    setSettings((s) => ({ ...s, ...partial }));
  }, []);

  const onTsl51PresetChange = useCallback((preset: Tsl51Preset) => {
    setTsl51Preset(preset);
    setSettings((s) => ({ ...s, ...TSL51_PRESETS[preset] }));
  }, []);

  const onFsPresetChange = useCallback((preset: FsPreset) => {
    setFsPreset(preset);
    setSettings((s) => ({ ...s, ...FS_PRESETS[preset] }));
  }, []);

  const onFsDynamicPresetChange = useCallback((preset: FsDynamicPreset) => {
    setFsDynamicPreset(preset);
    setSettings((s) => ({ ...s, ...FS_DYNAMIC_PRESETS[preset] }));
  }, []);

  const handleCameraError = useCallback((message: string) => {
    setStreaming(false);
    setError(message);
  }, []);

  const drawerProps = {
    tracks,
    track,
    onTrackChange: setTrack,
    artifacts,
    selectedArtifact,
    onArtifactChange: setSelectedArtifact,
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
    onLoad: handleLoad,
    loading,
    showSkeleton,
    onShowSkeletonChange: setShowSkeleton,
  };

  return (
    <main className="mx-auto min-h-screen max-w-[1600px] px-3 py-4 pb-12 sm:px-5 lg:px-6">
      <div className="app-shell overflow-hidden rounded-panel">

        {/* ── Header ───────────────────────────────────────────────── */}
        <header
          className="border-b border-line bg-panel px-5 py-5 sm:px-7 lg:px-8 animate-fade-rise"
          style={{ animationDelay: "0ms" }}
        >
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="max-w-2xl">
              {/* ONE eyebrow — the only .section-kicker in the app */}
              <p className="section-kicker">TSL Translation Console</p>
              <h1 className="mt-2 text-[clamp(1.75rem,1.2rem+2vw,2.75rem)] font-bold leading-tight tracking-[-0.01em] text-ink">
                TSL แปลภาษามือ
              </h1>
              <p className="mt-2 max-w-lg text-sm leading-7 text-subtle">
                โหลดโมเดล เปิดกล้อง และตรวจผลถอดความภาษามือไทยแบบเรียลไทม์
              </p>
            </div>

            {/* Status chips + theme toggle */}
            <div className="flex flex-col gap-3 lg:items-end">
              <div className="flex items-center gap-2">
                <div className="flex flex-wrap gap-2">
                  <StatusChip status={connectionStatus} label={connectionLabel} />
                  <StatusChip status={modelStatus} label={modelLabel} />
                  <StatusChip status="idle" label={selectedTrackTitle} />
                </div>
                <ThemeToggle />
              </div>

              {/* Meta row — Session / Artifact / FPS */}
              <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-subtle">
                <span>
                  Session{" "}
                  <span className="font-mono font-medium text-text">
                    {sessionId ? sessionId.slice(0, 8) : "—"}
                  </span>
                </span>
                <span className="min-w-0 max-w-[22ch] truncate">
                  Artifact{" "}
                  <span className="font-mono font-medium text-text">{artifactLabel}</span>
                </span>
                <span>
                  FPS{" "}
                  <span className="font-mono font-medium text-text">
                    {(state.prediction?.fps ?? 0).toFixed(1)}
                  </span>
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* ── Workspace ─────────────────────────────────────────────── */}
        <section className="bg-page px-4 py-5 sm:px-6 lg:px-8 xl:px-10">

          {/* Step indicator */}
          <div
            className="animate-fade-rise"
            style={{ animationDelay: "60ms" }}
          >
            <Stepper active={step} />
          </div>

          {/* Error banner */}
          {error && (
            <div className="mt-4 rounded-card border border-danger/30 bg-danger/10 px-4 py-3 text-sm font-medium text-danger shadow-sm">
              {error}
            </div>
          )}

          {/* Mobile drawer toggle — only visible below lg */}
          <div className="mt-4 lg:hidden">
            <button
              type="button"
              onClick={() => setShowDrawer((v) => !v)}
              className="flex w-full items-center justify-between rounded-field border border-line bg-panel px-4 py-2.5 text-sm font-semibold text-text shadow-card"
            >
              <span>⚙ ตั้งค่าระบบ — {selectedTrackTitle}</span>
              <span className="text-muted" aria-hidden>{showDrawer ? "▲" : "▼"}</span>
            </button>
          </div>

          {/* Main workspace grid */}
          <div className="mt-4 grid gap-5 lg:grid-cols-[260px_minmax(0,1fr)] xl:grid-cols-[280px_minmax(0,1fr)] 2xl:grid-cols-[300px_minmax(0,1fr)]">

            {/* Drawer — disclosure on mobile/tablet, sticky rail on >=lg */}
            <div
              className={`${showDrawer ? "block" : "hidden"} lg:block xl:self-start animate-fade-rise`}
              style={{ animationDelay: "100ms" }}
            >
              <ModelDrawer {...drawerProps} />
            </div>

            {/* Main content column */}
            <div
              className="min-w-0 space-y-4 animate-fade-rise"
              style={{ animationDelay: "140ms" }}
            >
              <ControlsBar
                streaming={streaming}
                modelLoaded={modelLoaded}
                onStart={() => {
                  setError(null);
                  setStreaming(true);
                }}
                onStop={() => setStreaming(false)}
                fps={state.prediction?.fps ?? 0}
                topkText={state.prediction?.topk_text ?? ""}
                confidence={state.prediction?.confidence ?? null}
                topk={state.prediction?.topk ?? []}
                bufferingProgress={state.bufferingProgress}
                track={track}
              />

              {/* Camera + Transcript */}
              <div
                className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(300px,0.8fr)] 2xl:grid-cols-[minmax(0,1.7fr)_minmax(320px,0.74fr)] animate-fade-rise"
                style={{ animationDelay: "180ms" }}
              >
                <CameraStage
                  streaming={streaming}
                  onReady={attachVideo}
                  onError={handleCameraError}
                  overlayLabel={state.displayLabel}
                  live={state.connected && streaming}
                  landmarks={state.landmarks}
                  handsDetected={state.handsDetected}
                  showSkeleton={showSkeleton}
                  predictionStatus={state.predictionStatus}
                />
                <TranscriptPanel
                  transcript={state.transcript}
                  pendingLabel={
                    state.prediction?.committed_label
                      ? glossLabel(state.prediction.committed_label)
                      : state.displayLabel
                  }
                  onSpace={() => handleTranscript("space")}
                  onBackspace={() => handleTranscript("backspace")}
                  onClear={() => handleTranscript("clear")}
                />
              </div>

              {/* Confidence chart */}
              <section
                className="rounded-panel border border-line bg-panel p-4 shadow-card sm:p-5 animate-fade-rise"
                style={{ animationDelay: "220ms" }}
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="text-base font-bold text-ink">แนวโน้มความมั่นใจ</h3>
                    <p className="mt-0.5 text-xs text-subtle">Signal confidence over time</p>
                  </div>
                  <StatusChip
                    status={state.hist.length ? "success" : "idle"}
                    label={state.hist.length ? `${state.hist.length} จุดข้อมูล` : "รอสัญญาณ"}
                  />
                </div>
                <div className="mt-4">
                  <ConfidenceChart data={state.hist} />
                </div>
              </section>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
