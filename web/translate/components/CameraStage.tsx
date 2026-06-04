"use client";

import { useEffect, useRef } from "react";
import {
  drawHolisticOverlay,
  hasAnyLandmarks,
  type HolisticLandmarks,
  type HandsDetected,
} from "@/lib/drawLandmarks";

interface Props {
  streaming: boolean;
  onReady: (video: HTMLVideoElement, canvas: HTMLCanvasElement) => void;
  onError: (message: string) => void;
  overlayLabel: string;
  live: boolean;
  landmarks: HolisticLandmarks | null;
  handsDetected: HandsDetected;
  showSkeleton: boolean;
  predictionStatus: string;
}

export function CameraStage({
  streaming,
  onReady,
  onError,
  overlayLabel,
  live,
  landmarks,
  handsDetected,
  showSkeleton,
  predictionStatus,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  const hasPrediction = Boolean(
    overlayLabel &&
      overlayLabel !== "?" &&
      !overlayLabel.startsWith("Unknown /"),
  );

  const noHandsDetected = live && !handsDetected.left && !handsDetected.right;
  const waitingLandmarks = live && showSkeleton && !hasAnyLandmarks(landmarks);

  // Camera stream — untouched (lib/drawLandmarks logic preserved)
  useEffect(() => {
    let stream: MediaStream | null = null;
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas) return;

    const start = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        video.srcObject = stream;
        await video.play();
        onReady(video, canvas);
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        onError(`เปิดกล้องไม่ได้: ${message}`);
      }
    };

    if (streaming) start();

    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    };
  }, [streaming, onReady, onError]);

  // Landmark overlay draw — untouched (drawHolisticOverlay tuned for dark bg)
  useEffect(() => {
    const video = videoRef.current;
    const overlay = overlayCanvasRef.current;
    if (!video || !overlay || !streaming || !showSkeleton) {
      if (overlay) {
        const ctx = overlay.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, overlay.width, overlay.height);
      }
      return;
    }

    const draw = () => {
      const w = video.clientWidth;
      const h = video.clientHeight;
      if (w < 2 || h < 2) return;
      if (overlay.width !== w || overlay.height !== h) {
        overlay.width = w;
        overlay.height = h;
      }
      const ctx = overlay.getContext("2d");
      if (!ctx) return;
      drawHolisticOverlay(ctx, landmarks, w, h, { mirror: true });
    };

    draw();
  }, [landmarks, streaming, showSkeleton]);

  const handBadge =
    streaming && live
      ? `มือซ้าย ${handsDetected.left ? "✓" : "✗"} · มือขวา ${handsDetected.right ? "✓" : "✗"}`
      : null;

  // Border state — crisp ring-2, no heavy ring-4 glow
  const containerBorder = live
    ? "border-success/50 ring-2 ring-success/20"
    : streaming
      ? "border-warning/50 ring-2 ring-warning/20"
      : "border-line";

  return (
    <div
      className={`relative overflow-hidden rounded-panel border bg-zinc-950 shadow-card transition-colors ${containerBorder}`}
    >
      {/* Top badge bar — floats over video, always dark-scrim style */}
      <div className="absolute inset-x-0 top-0 z-10 flex flex-wrap items-start justify-between gap-2 p-3">
        {/* Live / connecting / idle badge */}
        <span
          className={`rounded-full border px-3 py-1 text-xs font-bold backdrop-blur ${
            live
              ? "border-success/40 bg-black/55 text-success"
              : streaming
                ? "border-warning/40 bg-black/55 text-warning"
                : "border-white/15 bg-black/40 text-white/70"
          }`}
        >
          {live ? "LIVE · พร้อมทำนาย" : streaming ? "กำลังเชื่อมต่อกล้อง" : "รอเปิดกล้อง"}
        </span>

        {/* Right-side info pills */}
        <div className="flex max-w-[min(100%,28rem)] flex-wrap items-center justify-end gap-2">
          {handBadge && (
            <span
              className={`rounded-full border px-3 py-1 text-xs font-semibold backdrop-blur ${
                handsDetected.left || handsDetected.right
                  ? "border-success/35 bg-black/55 text-success"
                  : "border-warning/35 bg-black/55 text-warning"
              }`}
            >
              {handBadge}
            </span>
          )}
          {noHandsDetected && (
            <span className="rounded-full border border-warning/40 bg-black/60 px-3 py-1 text-[11px] font-medium leading-snug text-warning backdrop-blur">
              Server ไม่เห็นมือ — ถอยห่างให้เห็นไหล่+มือในเฟรม
            </span>
          )}
          {waitingLandmarks && (
            <span className="rounded-full border border-white/20 bg-black/50 px-3 py-1 text-xs font-semibold text-white/85 backdrop-blur">
              รอ skeleton…
            </span>
          )}
          {predictionStatus === "no_hand" && live && (
            <span className="rounded-full border border-danger/40 bg-black/60 px-3 py-1 text-xs font-semibold text-danger backdrop-blur">
              ไม่พบมือ (server)
            </span>
          )}
          <span className="rounded-full border border-white/15 bg-black/35 px-3 py-1 text-xs font-semibold text-white/70 backdrop-blur">
            16:9 preview
          </span>
        </div>
      </div>

      {/* Prediction label overlay — solid, no glass */}
      {hasPrediction && (
        <div className="absolute bottom-4 left-1/2 z-10 w-[min(92%,28rem)] -translate-x-1/2 rounded-card bg-brand px-5 py-3 text-center shadow-panel">
          <span className="text-[clamp(2rem,6vw,3rem)] font-bold leading-tight text-brand-fg">
            {overlayLabel}
          </span>
        </div>
      )}

      {/* Video + overlay canvas — mirrored for natural display (untouched) */}
      <div className="relative aspect-[16/10] min-h-[24rem] w-full xl:min-h-[34rem]">
        <video
          ref={videoRef}
          className="absolute inset-0 h-full w-full scale-x-[-1] bg-zinc-900 object-cover"
          playsInline
          muted
        />
        <canvas
          ref={overlayCanvasRef}
          className="pointer-events-none absolute inset-0 z-[5] h-full w-full"
          aria-hidden
        />
      </div>
      <canvas ref={captureCanvasRef} className="hidden" aria-hidden />

      {/* Idle overlay — themed (shows page/panel colors) */}
      {!streaming && (
        <div className="absolute inset-0 flex items-center justify-center bg-page/92 p-6 text-center backdrop-blur-sm">
          <div className="max-w-sm rounded-card border border-line bg-panel px-6 py-5 shadow-card">
            <p className="text-base font-bold text-ink">กล้องยังไม่เริ่มทำงาน</p>
            <p className="mt-2 text-sm leading-6 text-subtle">
              โหลดโมเดลให้พร้อม แล้วกดเริ่มกล้องเพื่อเข้าสู่โหมดแปลสด
            </p>
          </div>
        </div>
      )}

      {/* Connecting overlay */}
      {streaming && !live && (
        <div className="absolute inset-x-4 bottom-4 z-10 rounded-card border border-warning/30 bg-warning/10 px-4 py-3 text-sm text-warning backdrop-blur">
          กำลังรอสัญญาณภาพและ WebSocket — หากใช้ครั้งแรกให้อนุญาตสิทธิ์กล้องในเบราว์เซอร์
        </div>
      )}
    </div>
  );
}
