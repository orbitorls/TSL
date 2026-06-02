"use client";

interface Props {
  transcript: string;
  pendingLabel: string;
  onSpace: () => void;
  onBackspace: () => void;
  onClear: () => void;
}

export function TranscriptPanel({ transcript, pendingLabel, onSpace, onBackspace, onClear }: Props) {
  const copy = () => {
    if (transcript) navigator.clipboard.writeText(transcript);
  };

  return (
    <div className="flex h-full min-h-0 flex-col rounded-2xl border border-border bg-panel p-5 shadow-sm">
      <h2 className="text-lg font-bold text-brand">ข้อความที่แปลแล้ว</h2>
      <p className="mt-1 text-sm text-subtle">สะสมอัตโนมัติเมื่อโมเดลมั่นใจพอ</p>

      <div className="mt-4 min-h-[8rem] max-h-[16rem] flex-1 overflow-y-auto rounded-xl border border-dashed border-border bg-brand-ghost/40 p-4 md:max-h-[18rem] lg:max-h-[20rem]">
        <p className="break-words text-3xl font-bold leading-relaxed text-text [overflow-wrap:anywhere] md:text-4xl">
          {transcript || <span className="text-subtle">ยังไม่มีข้อความ</span>}
        </p>
        {pendingLabel && pendingLabel !== "?" && (
          <p className="mt-3 text-sm text-subtle">
            กำลังทาย: <span className="font-semibold text-brand-light">{pendingLabel}</span>
          </p>
        )}
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onSpace}
          className="rounded-xl border border-border bg-white px-4 py-2 text-sm font-medium hover:bg-brand-ghost"
        >
          เว้นวรรค
        </button>
        <button
          type="button"
          onClick={onBackspace}
          className="rounded-xl border border-border bg-white px-4 py-2 text-sm font-medium hover:bg-brand-ghost"
        >
          ลบตัวสุดท้าย
        </button>
        <button
          type="button"
          onClick={onClear}
          className="rounded-xl border border-border bg-white px-4 py-2 text-sm font-medium hover:bg-brand-ghost"
        >
          ล้าง
        </button>
        <button
          type="button"
          onClick={copy}
          disabled={!transcript}
          className="rounded-xl bg-brand px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          คัดลอก
        </button>
      </div>
    </div>
  );
}
