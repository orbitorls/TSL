"use client";

import { useState } from "react";
import { STATUS } from "@/lib/status";

interface Props {
  transcript: string;
  pendingLabel: string;
  onSpace: () => void;
  onBackspace: () => void;
  onClear: () => void;
}

export function TranscriptPanel({ transcript, pendingLabel, onSpace, onBackspace, onClear }: Props) {
  const hasTranscript = transcript.trim().length > 0;
  const hasPending = Boolean(pendingLabel && pendingLabel !== "?");
  const [copied, setCopied] = useState(false);

  const copy = () => {
    if (!transcript) return;
    navigator.clipboard.writeText(transcript).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  };

  // Derived status for the header chip
  const chipStatus = hasTranscript ? STATUS.success : hasPending ? STATUS.connecting : STATUS.idle;
  const chipLabel = hasTranscript ? "มีข้อความ" : hasPending ? "กำลังทาย" : "ว่าง";

  return (
    <div className="flex h-full min-h-0 flex-col rounded-panel border border-line bg-panel p-5 shadow-card">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-bold text-ink">ข้อความที่แปลแล้ว</h2>
          <p className="mt-1 text-xs leading-6 text-subtle">
            สะสมอัตโนมัติเมื่อโมเดลมั่นใจพอ แก้ไขด้วยปุ่มด้านล่างได้
          </p>
        </div>
        <span
          className={`shrink-0 rounded-full border px-3 py-1 text-xs font-semibold ${chipStatus.border} ${chipStatus.bg} ${chipStatus.text}`}
        >
          {chipLabel}
        </span>
      </div>

      {/* Transcript well */}
      <div className="mt-4 min-h-[10rem] max-h-[16rem] flex-1 overflow-y-auto rounded-card border border-line bg-panel-2 p-4 md:max-h-[18rem] lg:max-h-[20rem]">
        {hasTranscript ? (
          <p className="break-words text-[clamp(1.5rem,3vw,2.25rem)] font-bold leading-relaxed text-ink [overflow-wrap:anywhere]">
            {transcript}
          </p>
        ) : (
          <div className="flex min-h-[7rem] flex-col justify-center text-center">
            <p className="text-2xl font-bold text-muted">ยังไม่มีข้อความ</p>
            <p className="mt-2 text-xs leading-6 text-muted">
              เมื่อท่ามือผ่านเกณฑ์ความมั่นใจ ข้อความจะแสดงที่นี่
            </p>
          </div>
        )}

        {hasPending && (
          <div className="mt-4 rounded-card border border-brand/20 bg-brand-soft px-3 py-2 text-sm text-subtle">
            กำลังทาย:{" "}
            <span className="font-bold text-brand-strong">{pendingLabel}</span>
          </div>
        )}
      </div>

      {/* Action buttons */}
      <div className="mt-4 grid grid-cols-2 gap-2 sm:flex sm:flex-wrap">
        <button
          type="button"
          onClick={onSpace}
          className="rounded-field border border-line bg-panel px-4 py-2 text-sm font-semibold text-text hover:bg-panel-2 disabled:cursor-not-allowed disabled:opacity-40"
        >
          เว้นวรรค
        </button>
        <button
          type="button"
          onClick={onBackspace}
          disabled={!hasTranscript}
          className="rounded-field border border-line bg-panel px-4 py-2 text-sm font-semibold text-text hover:bg-panel-2 disabled:cursor-not-allowed disabled:opacity-40"
        >
          ลบตัวสุดท้าย
        </button>
        <button
          type="button"
          onClick={onClear}
          disabled={!hasTranscript}
          className="rounded-field border border-line bg-panel px-4 py-2 text-sm font-semibold text-text hover:bg-panel-2 disabled:cursor-not-allowed disabled:opacity-40"
        >
          ล้าง
        </button>
        <button
          type="button"
          onClick={copy}
          disabled={!hasTranscript}
          className={`rounded-field px-4 py-2 text-sm font-semibold shadow-card transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
            copied
              ? "bg-success text-panel"
              : "bg-brand text-brand-fg hover:bg-brand-strong"
          }`}
        >
          {copied ? "คัดลอกแล้ว ✓" : "คัดลอก"}
        </button>
      </div>
    </div>
  );
}
