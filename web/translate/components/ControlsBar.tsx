"use client";

interface Props {
  streaming: boolean;
  modelLoaded: boolean;
  onStart: () => void;
  onStop: () => void;
  fps: number;
  topkText: string;
}

export function ControlsBar({ streaming, modelLoaded, onStart, onStop, fps, topkText }: Props) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-border bg-panel px-4 py-3 shadow-sm">
      <div className="flex gap-2">
        <button
          type="button"
          onClick={onStart}
          disabled={!modelLoaded || streaming}
          className="rounded-xl bg-brand px-5 py-2 font-medium text-white disabled:opacity-40"
        >
          เริ่มกล้อง
        </button>
        <button
          type="button"
          onClick={onStop}
          disabled={!streaming}
          className="rounded-xl border border-border bg-white px-5 py-2 font-medium disabled:opacity-40"
        >
          หยุดกล้อง
        </button>
      </div>
      <div className="text-sm text-subtle">
        FPS: <strong className="text-text">{fps.toFixed(1)}</strong>
        {topkText && (
          <span className="ml-4 hidden sm:inline">
            Top-K: <strong className="text-text">{topkText}</strong>
          </span>
        )}
      </div>
    </div>
  );
}
