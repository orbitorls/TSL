import { STATUS, type Status } from "@/lib/status";

interface DotProps {
  status: Status;
  className?: string;
}

/** A small colored indicator dot. The "live" status adds a pulse ring animation. */
export function StatusDot({ status, className = "" }: DotProps) {
  const s = STATUS[status];
  return (
    <span
      className={`inline-block h-2 w-2 shrink-0 rounded-full ${s.dot} ${
        status === "live" ? "animate-pulse-live" : ""
      } ${className}`}
      aria-hidden
    />
  );
}

interface ChipProps {
  status: Status;
  label: string;
  /** Extra classes on the outer pill */
  className?: string;
}

/**
 * A status pill — colored dot + label text — themed automatically.
 * Replaces the old .status-chip / .status-dot CSS classes.
 */
export function StatusChip({ status, label, className = "" }: ChipProps) {
  const s = STATUS[status];
  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold ${s.bg} ${s.border} ${className}`}
    >
      <StatusDot status={status} />
      <span className={s.text}>{label}</span>
    </div>
  );
}
