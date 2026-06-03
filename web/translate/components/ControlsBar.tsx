"use client";

import type { TopKEntry } from "@/lib/types";
import { glossLabel } from "@/lib/gloss";

interface Props {
  streaming: boolean;
  modelLoaded: boolean;
  onStart: () => void;
  onStop: () => void;
  fps: number;
  topkText: string;
  confidence: number | null;
  topk: TopKEntry[];
  bufferingProgress: string | null;
}

function formatBufferingProgress(progress: string): string {
  const [current, total] = progress.split("/");
  if (!current || !total) return "กำลังเตรียม";
  return `กำลังเตรียม ${current}/${total}`;
}

function formatTopkSummary(topk: TopKEntry[]): string {
  if (!topk.length) return "รอข้อมูล";
  return topk
    .slice(0, 3)
    .map((entry) => `${glossLabel(entry.label)} ${(entry.p * 100).toFixed(0)}%`)
    .join(" · ");
}

function formatMargin(topk: TopKEntry[]): string | null {
  if (topk.length < 2) return null;
  const margin = topk[0].p - topk[1].p;
  return margin.toFixed(2);
}

export function ControlsBar({
  streaming,
  modelLoaded,
  onStart,
  onStop,
  fps,
  topkText,
  confidence,
  topk,
  bufferingProgress,
}: Props) {
  const statusText = streaming ? "กล้องกำลังทำงาน" : modelLoaded ? "พร้อมเปิดกล้อง" : "รอโหลดโมเดล";
  const margin = formatMargin(topk);
  const topkSummary = topk.length ? formatTopkSummary(topk) : topkText || "รอข้อมูล";

  return (
    <div className="flex flex-col gap-3 rounded-3xl border border-border bg-panel/95 px-4 py-3 shadow-sm ring-1 ring-white/60 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              streaming ? "bg-emerald-500 shadow-[0_0_0_4px_rgba(16,185,129,0.16)]" : modelLoaded ? "bg-amber-400" : "bg-zinc-300"
            }`}
            aria-hidden
          />
          <p className="text-sm font-bold text-text">{statusText}</p>
        </div>
        <p className="mt-0.5 text-xs text-subtle">
          {streaming ? "ทำท่าให้ครบแล้วหยุดมือสั้นๆ เพื่อ commit คำ (โหมดแม่นยำ)" : "โหลดโมเดลแล้วจึงเริ่มกล้องเพื่อแปลภาษามือ"}
        </p>
      </div>

      <div className="flex flex-col gap-3 sm:items-end">
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onStart}
            disabled={!modelLoaded || streaming}
            className="rounded-xl bg-brand px-5 py-2 font-semibold text-white shadow-sm transition hover:bg-brand/95 disabled:cursor-not-allowed disabled:opacity-40"
          >
            เริ่มกล้อง
          </button>
          <button
            type="button"
            onClick={onStop}
            disabled={!streaming}
            className="rounded-xl border border-border bg-white px-5 py-2 font-semibold text-text transition hover:bg-brand-ghost disabled:cursor-not-allowed disabled:opacity-40"
          >
            หยุดกล้อง
          </button>
        </div>

        <div className="flex flex-wrap gap-2 text-xs text-subtle">
          {streaming && bufferingProgress && (
            <span className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-amber-800">
              {formatBufferingProgress(bufferingProgress)}
            </span>
          )}
          <span className="rounded-full border border-border bg-page px-3 py-1">
            FPS <strong className="ml-1 text-text">{fps.toFixed(1)}</strong>
          </span>
          {confidence != null && (
            <span className="rounded-full border border-border bg-page px-3 py-1">
              ความมั่นใจ <strong className="ml-1 text-text">{(confidence * 100).toFixed(0)}%</strong>
            </span>
          )}
          {margin != null && (
            <span className="rounded-full border border-border bg-page px-3 py-1">
              Margin <strong className="ml-1 text-text">{margin}</strong>
            </span>
          )}
          <span className="max-w-[min(100%,28rem)] rounded-full border border-border bg-page px-3 py-1">
            Top-3 <strong className="ml-1 text-text">{topkSummary}</strong>
          </span>
        </div>
      </div>
    </div>
  );
}
