"use client";

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

  const copy = () => {
    if (transcript) navigator.clipboard.writeText(transcript);
  };

  return (
    <div className="flex h-full min-h-0 flex-col rounded-3xl border border-border bg-panel/95 p-5 shadow-sm ring-1 ring-white/60">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-brand-light">Transcript</p>
          <h2 className="mt-1 text-xl font-bold text-brand">ข้อความที่แปลแล้ว</h2>
          <p className="mt-1 text-sm leading-6 text-subtle">สะสมอัตโนมัติเมื่อโมเดลมั่นใจพอ และยังแก้ไขด้วยปุ่มด้านล่างได้</p>
        </div>
        <span
          className={`shrink-0 rounded-full px-3 py-1 text-xs font-semibold ${
            hasTranscript ? "bg-emerald-50 text-emerald-700" : hasPending ? "bg-amber-50 text-amber-700" : "bg-brand-ghost text-subtle"
          }`}
        >
          {hasTranscript ? "มีข้อความ" : hasPending ? "กำลังทาย" : "ว่าง"}
        </span>
      </div>

      <div className="mt-4 min-h-[10rem] max-h-[16rem] flex-1 overflow-y-auto rounded-2xl border border-dashed border-border bg-brand-ghost/40 p-4 md:max-h-[18rem] lg:max-h-[20rem]">
        {hasTranscript ? (
          <p className="break-words text-3xl font-bold leading-relaxed text-text [overflow-wrap:anywhere] md:text-4xl">{transcript}</p>
        ) : (
          <div className="flex min-h-[7rem] flex-col justify-center text-center">
            <p className="text-2xl font-bold text-text/45">ยังไม่มีข้อความ</p>
            <p className="mt-2 text-sm leading-6 text-subtle">เมื่อท่ามือผ่านเกณฑ์ความมั่นใจ ข้อความจะแสดงที่นี่</p>
          </div>
        )}

        {hasPending && (
          <div className="mt-4 rounded-2xl border border-brand/15 bg-white/80 px-3 py-2 text-sm text-subtle">
            กำลังทาย: <span className="font-bold text-brand-light">{pendingLabel}</span>
          </div>
        )}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 sm:flex sm:flex-wrap">
        <button
          type="button"
          onClick={onSpace}
          className="rounded-xl border border-border bg-white px-4 py-2 text-sm font-semibold text-text transition hover:bg-brand-ghost"
        >
          เว้นวรรค
        </button>
        <button
          type="button"
          onClick={onBackspace}
          disabled={!hasTranscript}
          className="rounded-xl border border-border bg-white px-4 py-2 text-sm font-semibold text-text transition hover:bg-brand-ghost disabled:cursor-not-allowed disabled:opacity-40"
        >
          ลบตัวสุดท้าย
        </button>
        <button
          type="button"
          onClick={onClear}
          disabled={!hasTranscript}
          className="rounded-xl border border-border bg-white px-4 py-2 text-sm font-semibold text-text transition hover:bg-brand-ghost disabled:cursor-not-allowed disabled:opacity-40"
        >
          ล้าง
        </button>
        <button
          type="button"
          onClick={copy}
          disabled={!hasTranscript}
          className="rounded-xl bg-brand px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-brand/95 disabled:cursor-not-allowed disabled:opacity-40"
        >
          คัดลอก
        </button>
      </div>
    </div>
  );
}
