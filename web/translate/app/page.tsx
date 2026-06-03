"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { CameraStage } from "@/components/CameraStage";
import { ConfidenceChart } from "@/components/ConfidenceChart";
import { ControlsBar } from "@/components/ControlsBar";
import { ModelDrawer } from "@/components/ModelDrawer";
import { Stepper } from "@/components/Stepper";
import { TranscriptPanel } from "@/components/TranscriptPanel";
import {
  createSession,
  fetchArtifacts,
  fetchTracks,
  loadModel,
  patchSettings,
  transcriptAction,
} from "@/lib/api";
import type { Artifact, InferenceSettings, Track, TrackKey, Tsl51Preset } from "@/lib/types";
import { TSL51_PRESETS } from "@/lib/types";
import { useInferenceWs } from "@/lib/ws";
import { glossLabel } from "@/lib/gloss";

const defaultSettings = (track: TrackKey): InferenceSettings => ({
  threshold: track === "fingerspelling" ? 0.7 : TSL51_PRESETS.accurate.threshold,
  alpha: 0.4,
  top_k: 3,
  motion_min: 0.008,
  ...(track === "tsl51" ? TSL51_PRESETS.accurate : {}),
});

export default function TranslatePage() {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [track, setTrack] = useState<TrackKey>("fingerspelling");
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [selectedArtifact, setSelectedArtifact] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [settings, setSettings] = useState<InferenceSettings>(defaultSettings("fingerspelling"));
  const [tsl51Preset, setTsl51Preset] = useState<Tsl51Preset>("accurate");
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelInfo, setModelInfo] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    setTsl51Preset("accurate");
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

  const handleCameraError = useCallback((message: string) => {
    setStreaming(false);
    setError(message);
  }, []);

  return (
    <main className="mx-auto min-h-screen max-w-[1800px] px-3 py-5 pb-12 sm:px-5 lg:px-6">
      <div className="operator-shell overflow-hidden rounded-[2rem]">
        <header className="border-b border-line bg-panel/88 px-5 py-5 sm:px-7 lg:px-8">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="section-kicker">TSL Translation Console</div>
              <h1 className="mt-2 text-3xl font-bold leading-tight text-ink md:text-4xl">TSL แปลภาษามือ</h1>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-subtle md:text-base">
                พื้นที่ปฏิบัติงานสำหรับโหลดโมเดล เปิดกล้อง และตรวจผลถอดความภาษามือไทยแบบเรียลไทม์
              </p>
            </div>

            <div className="grid gap-2 sm:grid-cols-3 lg:min-w-[520px]">
              <div className="status-chip">
                <span className={`status-dot ${streaming ? "bg-success" : "bg-muted"}`} />
                {connectionLabel}
              </div>
              <div className="status-chip">
                <span className={`status-dot ${modelLoaded ? "bg-success" : selectedArtifact ? "bg-warning" : "bg-muted"}`} />
                {modelLabel}
              </div>
              <div className="status-chip min-w-0" title={artifactLabel}>
                <span className="status-dot bg-brand" />
                <span className="truncate">{selectedTrackTitle}</span>
              </div>
            </div>
          </div>

          <div className="mt-5 grid gap-3 border-t border-line pt-4 text-xs text-subtle sm:grid-cols-3">
            <div>
              <span className="font-semibold text-ink">Session</span> {sessionId ? sessionId.slice(0, 8) : "กำลังเตรียม"}
            </div>
            <div className="min-w-0">
              <span className="font-semibold text-ink">Artifact</span>{" "}
              <span className="truncate align-bottom">{artifactLabel}</span>
            </div>
            <div>
              <span className="font-semibold text-ink">FPS</span> {(state.prediction?.fps ?? 0).toFixed(1)}
            </div>
          </div>
        </header>

        <section className="bg-page/55 px-4 py-5 sm:px-6 lg:px-8 xl:px-10">
          <Stepper active={step} />

          {error && (
            <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-800 shadow-sm">
              {error}
            </div>
          )}

          <div className="mt-5 grid gap-5 xl:grid-cols-[260px_minmax(0,1fr)] 2xl:grid-cols-[280px_minmax(0,1fr)]">
            <div className="xl:self-start">
              <ModelDrawer
                tracks={tracks}
                track={track}
                onTrackChange={setTrack}
                artifacts={artifacts}
                selectedArtifact={selectedArtifact}
                onArtifactChange={setSelectedArtifact}
                settings={settings}
                onSettingsChange={onSettingsChange}
                tsl51Preset={tsl51Preset}
                onTsl51PresetChange={onTsl51PresetChange}
                modelLoaded={modelLoaded}
                modelInfo={modelInfo}
                onLoad={handleLoad}
                loading={loading}
              />
            </div>

            <div className="min-w-0 space-y-5">
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
              />

              <div className="grid gap-5 2xl:grid-cols-[minmax(0,1.7fr)_minmax(320px,0.72fr)]">
                <CameraStage
                  streaming={streaming}
                  onReady={attachVideo}
                  onError={handleCameraError}
                  overlayLabel={state.displayLabel}
                  live={state.connected && streaming}
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

              <section className="workspace-panel rounded-[1.35rem] p-4 sm:p-5">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="section-kicker">Signal confidence</p>
                    <h3 className="mt-1 text-lg font-bold text-brand">แนวโน้มความมั่นใจ</h3>
                  </div>
                  <div className="status-chip">
                    <span className={`status-dot ${state.hist.length ? "bg-success" : "bg-muted"}`} />
                    {state.hist.length ? `${state.hist.length} จุดข้อมูล` : "รอสัญญาณ"}
                  </div>
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
