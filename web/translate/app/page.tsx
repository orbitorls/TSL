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
import type { Artifact, InferenceSettings, Track, TrackKey } from "@/lib/types";
import { useInferenceWs } from "@/lib/ws";

const defaultSettings = (track: TrackKey): InferenceSettings => ({
  threshold: track === "fingerspelling" ? 0.7 : 0.55,
  alpha: 0.4,
  top_k: 3,
  motion_min: 0.008,
});

export default function TranslatePage() {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [track, setTrack] = useState<TrackKey>("fingerspelling");
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [selectedArtifact, setSelectedArtifact] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [settings, setSettings] = useState<InferenceSettings>(defaultSettings("fingerspelling"));
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

  useEffect(() => {
    fetchTracks()
      .then(setTracks)
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    setSettings(defaultSettings(track));
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

  const handleCameraError = useCallback((message: string) => {
    setStreaming(false);
    setError(message);
  }, []);

  return (
    <main className="mx-auto max-w-6xl px-4 py-6 pb-16">
      <header className="mb-6 rounded-2xl border border-border bg-white/80 p-5 shadow-sm backdrop-blur">
        <h1 className="text-2xl font-bold text-brand md:text-3xl">TSL แปลภาษามือ</h1>
        <p className="mt-1 text-subtle">
          ระบบแปลภาษามือไทยเป็นข้อความแบบเรียลไทม์ · เลือกโมเดล · เปิดกล้อง · อ่านผลถอดความ
        </p>
      </header>

      <Stepper active={step} />

      {error && (
        <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          {error}
        </div>
      )}

      <div className="mt-6 grid gap-6 lg:grid-cols-[280px_1fr]">
        <ModelDrawer
          tracks={tracks}
          track={track}
          onTrackChange={setTrack}
          artifacts={artifacts}
          selectedArtifact={selectedArtifact}
          onArtifactChange={setSelectedArtifact}
          settings={settings}
          onSettingsChange={onSettingsChange}
          modelLoaded={modelLoaded}
          modelInfo={modelInfo}
          onLoad={handleLoad}
          loading={loading}
        />

        <div className="space-y-4">
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
          />

          <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
            <CameraStage
              streaming={streaming}
              onReady={attachVideo}
              onError={handleCameraError}
              overlayLabel={state.pendingLabel}
              live={state.connected && streaming}
            />
            <TranscriptPanel
              transcript={state.transcript}
              pendingLabel={state.pendingLabel}
              onSpace={() => handleTranscript("space")}
              onBackspace={() => handleTranscript("backspace")}
              onClear={() => handleTranscript("clear")}
            />
          </div>

          <section className="rounded-2xl border border-border bg-panel p-4 shadow-sm">
            <h3 className="font-bold text-brand">แนวโน้มความมั่นใจ</h3>
            <div className="mt-2">
              <ConfidenceChart data={state.hist} />
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
