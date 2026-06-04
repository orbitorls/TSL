"use client";

import type { TopKEntry, TrackKey } from "@/lib/types";
import { glossLabel } from "@/lib/gloss";
import { STATUS } from "@/lib/status";

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
  track?: TrackKey;
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

function fingerspellingAmbiguousHint(topk: TopKEntry[]): string | null {
  if (topk.length < 2) return null;
  const margin = topk[0].p - topk[1].p;
  if (margin >= 0.12) return null;
  const top2 = new Set(topk.slice(0, 2).map((e) => e.label));
  if (top2.has("KO_KAI") && top2.has("BOR_BAI_MAI")) {
    return "ท่าใกล้กัน (ก/บ) — ชูนิ้วให้ชัด / ถือมือนิ่ง";
  }
  return "ท่าใกล้กัน — ชูนิ้วให้ชัด / ถือมือนิ่ง";
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
  track,
}: Props) {
  const statusText = streaming
    ? "กล้องกำลังทำงาน"
    : modelLoaded
      ? "พร้อมเปิดกล้อง"
      : "รอโหลดโมเดล";

  const liveStatus = streaming ? STATUS.live : modelLoaded ? STATUS.connecting : STATUS.idle;
  const margin = formatMargin(topk);
  const fsHint =
    streaming && (track === "fingerspelling" || track === "fingerspelling_dynamic")
      ? fingerspellingAmbiguousHint(topk)
      : null;
  const topkSummary = topk.length ? formatTopkSummary(topk) : topkText || "รอข้อมูล";

  return (
    <div className="flex flex-col gap-3 rounded-panel border border-line bg-panel px-4 py-3.5 shadow-card sm:flex-row sm:items-center sm:justify-between">
      {/* Status + context hint */}
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`h-2.5 w-2.5 rounded-full ${liveStatus.dot} ${streaming ? "animate-pulse-live" : ""}`}
            aria-hidden
          />
          <p className="text-sm font-bold text-ink">{statusText}</p>
        </div>
        <p className="mt-0.5 text-xs leading-6 text-subtle">
          {streaming
            ? track === "fingerspelling"
              ? "ถือท่าให้นิ่ง 2–3 เฟรม · ท่าไม่ชัดจะไม่เดาตัวอักษร"
              : track === "fingerspelling_dynamic"
                ? "ทำท่าสองจังหวะให้ครบแล้วหยุดมือสั้นๆ · โหมดจังหวะ"
                : "ทำท่าให้ครบแล้วหยุดมือสั้นๆ เพื่อ commit คำ (โหมดแม่นยำ)"
            : "โหลดโมเดลแล้วจึงเริ่มกล้องเพื่อแปลภาษามือ"}
        </p>
        {fsHint && (
          <p className={`mt-1.5 rounded-field border px-2.5 py-1.5 text-xs font-medium ${STATUS.warning.border} ${STATUS.warning.bg} ${STATUS.warning.text}`}>
            {fsHint}
          </p>
        )}
      </div>

      {/* Buttons + metric pills */}
      <div className="flex flex-col gap-3 sm:items-end">
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onStart}
            disabled={!modelLoaded || streaming}
            className="rounded-field bg-brand px-5 py-2 text-sm font-semibold text-brand-fg shadow-card hover:bg-brand-strong disabled:cursor-not-allowed disabled:opacity-40"
          >
            เริ่มกล้อง
          </button>
          <button
            type="button"
            onClick={onStop}
            disabled={!streaming}
            className="rounded-field border border-line bg-panel px-5 py-2 text-sm font-semibold text-text hover:bg-panel-2 disabled:cursor-not-allowed disabled:opacity-40"
          >
            หยุดกล้อง
          </button>
        </div>

        {/* Metric pills */}
        <div className="flex flex-wrap gap-1.5 text-xs text-subtle">
          {streaming && bufferingProgress && (
            <span className={`rounded-full border px-3 py-1 font-medium ${STATUS.connecting.border} ${STATUS.connecting.bg} ${STATUS.connecting.text}`}>
              {formatBufferingProgress(bufferingProgress)}
            </span>
          )}
          <span className="rounded-full border border-line bg-panel-2 px-3 py-1">
            FPS{" "}
            <strong className="font-mono font-medium text-text">{fps.toFixed(1)}</strong>
          </span>
          {confidence != null && (
            <span className="rounded-full border border-line bg-panel-2 px-3 py-1">
              มั่นใจ{" "}
              <strong className="font-mono font-medium text-text">{(confidence * 100).toFixed(0)}%</strong>
            </span>
          )}
          {margin != null && (
            <span className="rounded-full border border-line bg-panel-2 px-3 py-1">
              Margin{" "}
              <strong className="font-mono font-medium text-text">{margin}</strong>
            </span>
          )}
          <span className="max-w-[min(100%,28rem)] rounded-full border border-line bg-panel-2 px-3 py-1">
            Top-3{" "}
            <strong className="ml-1 font-medium text-text">{topkSummary}</strong>
          </span>
        </div>
      </div>
    </div>
  );
}
