"use client";

import { useEffect, useRef } from "react";

interface Props {
  streaming: boolean;
  onReady: (video: HTMLVideoElement, canvas: HTMLCanvasElement) => void;
  onError: (message: string) => void;
  overlayLabel: string;
  live: boolean;
}

export function CameraStage({ streaming, onReady, onError, overlayLabel, live }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hasPrediction = Boolean(
    overlayLabel &&
      overlayLabel !== "?" &&
      !overlayLabel.startsWith("Unknown /"),
  );

  useEffect(() => {
    let stream: MediaStream | null = null;
    const video = videoRef.current;
    const canvas = canvasRef.current;
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

    if (streaming) {
      start();
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
      video.srcObject = null;
    };
  }, [streaming, onReady, onError]);

  return (
    <div
      className={`relative overflow-hidden rounded-3xl border bg-zinc-950 shadow-sm transition ${
        live ? "border-emerald-300 ring-4 ring-emerald-100" : streaming ? "border-amber-300 ring-4 ring-amber-100" : "border-border"
      }`}
    >
      <div className="absolute inset-x-0 top-0 z-10 flex flex-wrap items-start justify-between gap-2 p-3">
        <span
          className={`rounded-full border px-3 py-1 text-xs font-bold shadow-sm backdrop-blur ${
            live
              ? "border-emerald-300 bg-emerald-50/95 text-emerald-700"
              : streaming
                ? "border-amber-300 bg-amber-50/95 text-amber-700"
                : "border-border bg-white/95 text-subtle"
          }`}
        >
          {live ? "LIVE · พร้อมทำนาย" : streaming ? "กำลังเชื่อมต่อกล้อง" : "รอเปิดกล้อง"}
        </span>
        <span className="rounded-full border border-white/20 bg-black/35 px-3 py-1 text-xs font-semibold text-white backdrop-blur">
          16:9 preview
        </span>
      </div>

      {hasPrediction && (
        <div className="absolute bottom-4 left-1/2 z-10 w-[min(92%,28rem)] -translate-x-1/2 rounded-3xl border border-white/20 bg-brand/95 px-5 py-3 text-center text-4xl font-bold leading-tight text-white shadow-2xl backdrop-blur md:text-5xl">
          {overlayLabel}
        </div>
      )}

      <video
        ref={videoRef}
        className="aspect-[16/10] min-h-[24rem] w-full scale-x-[-1] bg-zinc-900 object-cover xl:min-h-[34rem]"
        playsInline
        muted
      />
      <canvas ref={canvasRef} className="hidden" aria-hidden />

      {!streaming && (
        <div className="absolute inset-0 flex items-center justify-center bg-page/90 p-6 text-center backdrop-blur-sm">
          <div className="max-w-sm rounded-3xl border border-border bg-white/90 px-6 py-5 shadow-sm">
            <p className="text-lg font-bold text-brand">กล้องยังไม่เริ่มทำงาน</p>
            <p className="mt-2 text-sm leading-6 text-subtle">โหลดโมเดลให้พร้อม แล้วกดเริ่มกล้องเพื่อเข้าสู่โหมดแปลสด</p>
          </div>
        </div>
      )}

      {streaming && !live && (
        <div className="absolute inset-x-4 bottom-4 z-10 rounded-2xl border border-amber-200 bg-amber-50/95 px-4 py-3 text-sm text-amber-800 shadow-sm backdrop-blur">
          กำลังรอสัญญาณภาพและ WebSocket หากใช้ครั้งแรกให้อนุญาตสิทธิ์กล้องในเบราว์เซอร์
        </div>
      )}
    </div>
  );
}
