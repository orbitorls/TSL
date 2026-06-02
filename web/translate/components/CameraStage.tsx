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
    <div className="relative overflow-hidden rounded-2xl border border-border bg-black/5 shadow-sm">
      <div className="absolute left-3 top-3 z-10 flex gap-2">
        {live ? (
          <span className="rounded-full border border-emerald-300 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700">
            LIVE
          </span>
        ) : (
          <span className="rounded-full border border-border bg-white/90 px-3 py-1 text-xs font-semibold text-subtle">
            รอเปิดกล้อง
          </span>
        )}
      </div>
      {overlayLabel && (
        <div className="absolute bottom-4 left-1/2 z-10 -translate-x-1/2 rounded-2xl bg-brand/90 px-6 py-3 text-4xl font-bold text-white shadow-lg">
          {overlayLabel}
        </div>
      )}
      <video ref={videoRef} className="aspect-video w-full bg-zinc-900 object-cover" playsInline muted />
      <canvas ref={canvasRef} className="hidden" aria-hidden />
      {!streaming && (
        <div className="absolute inset-0 flex items-center justify-center bg-page/80 text-subtle">
          กดเริ่มกล้องหลังโหลดโมเดล
        </div>
      )}
    </div>
  );
}
