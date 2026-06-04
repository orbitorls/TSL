"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { HandsDetected, PredictionMessage, TranscriptMessage, WsMessage } from "./types";
import { parseHolisticLandmarks, type HolisticLandmarks } from "./drawLandmarks";
import { getWsUrl } from "./api";
import { glossLabel } from "./gloss";

export interface LiveState {
  connected: boolean;
  prediction: PredictionMessage | null;
  transcript: string;
  displayLabel: string;
  bufferingProgress: string | null;
  hist: number[];
  error: string | null;
  landmarks: HolisticLandmarks | null;
  handsDetected: HandsDetected;
  predictionStatus: string;
}

const initial: LiveState = {
  connected: false,
  prediction: null,
  transcript: "",
  displayLabel: "",
  bufferingProgress: null,
  hist: [],
  error: null,
  landmarks: null,
  handsDetected: { left: false, right: false },
  predictionStatus: "ready",
};

const SYSTEM_LABEL_PREFIXES = ["กำลังเก็บเฟรม", "กำลังเซ็น", "ไม่พบมือ"] as const;

function isSystemLabel(label: string): boolean {
  return SYSTEM_LABEL_PREFIXES.some((prefix) => label.startsWith(prefix));
}

function derivePredictionUi(msg: PredictionMessage): Pick<LiveState, "displayLabel" | "bufferingProgress"> {
  const showPrediction = ["previewing", "predicted", "low_confidence"].includes(msg.status);
  const rawLabel =
    showPrediction && msg.label && !isSystemLabel(msg.label) ? msg.label : "";
  const displayLabel = rawLabel ? glossLabel(rawLabel) : "";
  return {
    displayLabel,
    bufferingProgress: msg.buffering,
  };
}

export function useInferenceWs(sessionId: string | null, streaming: boolean) {
  const wsRef = useRef<WebSocket | null>(null);
  const [state, setState] = useState<LiveState>(initial);
  const frameLoopRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const attachVideo = useCallback((video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
    videoRef.current = video;
    canvasRef.current = canvas;
  }, []);

  useEffect(() => {
    if (!sessionId || !streaming) {
      if (wsRef.current) {
        if (wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: "stop" }));
        }
        wsRef.current.close();
        wsRef.current = null;
      }
      if (frameLoopRef.current) {
        cancelAnimationFrame(frameLoopRef.current);
        frameLoopRef.current = null;
      }
      setState((s) => ({ ...s, connected: false }));
      return;
    }

    const ws = new WebSocket(getWsUrl(sessionId));
    wsRef.current = ws;

    ws.onopen = () => {
      setState((s) => ({ ...s, connected: true, error: null }));
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data as string) as WsMessage;
        if (msg.type === "prediction") {
          const ui = derivePredictionUi(msg);
          const landmarks = parseHolisticLandmarks(msg.landmarks);
          const handsDetected = msg.hands_detected ?? { left: false, right: false };
          setState((s) => ({
            ...s,
            prediction: msg,
            displayLabel: ui.displayLabel,
            bufferingProgress: ui.bufferingProgress,
            hist: msg.confidence_hist,
            transcript: msg.transcript || s.transcript,
            landmarks,
            handsDetected,
            predictionStatus: msg.status,
          }));
        } else if (msg.type === "transcript") {
          const t = msg as TranscriptMessage;
          setState((s) => ({ ...s, transcript: t.text }));
        } else if (msg.type === "error") {
          setState((s) => ({ ...s, error: msg.message }));
        }
      } catch {
        /* ignore */
      }
    };

    ws.onerror = () => {
      setState((s) => ({ ...s, error: "การเชื่อมต่อ WebSocket ล้มเหลว" }));
    };

    ws.onclose = () => {
      setState((s) => ({ ...s, connected: false }));
    };

    const sendFrame = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || ws.readyState !== WebSocket.OPEN || video.readyState < 2) {
        frameLoopRef.current = requestAnimationFrame(sendFrame);
        return;
      }
      const sourceW = video.videoWidth || 640;
      const sourceH = video.videoHeight || 480;
      const w = Math.min(640, sourceW);
      const h = Math.round((sourceH / sourceW) * w);
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        // Video is mirrored for display only; capture unflipped pixels for inference.
        ctx.drawImage(video, 0, 0, w, h);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.75);
        ws.send(JSON.stringify({ type: "frame", jpeg_base64: dataUrl }));
      }
      frameLoopRef.current = requestAnimationFrame(() => {
        setTimeout(() => sendFrame(), 66);
      });
    };

    frameLoopRef.current = requestAnimationFrame(sendFrame);

    return () => {
      if (frameLoopRef.current) cancelAnimationFrame(frameLoopRef.current);
      ws.close();
      wsRef.current = null;
    };
  }, [sessionId, streaming]);

  const setTranscript = useCallback((text: string) => {
    setState((s) => ({ ...s, transcript: text }));
  }, []);

  return { state, attachVideo, setTranscript };
}
