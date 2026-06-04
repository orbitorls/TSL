// Single source of truth for all status colors.
// IMPORTANT: All class names are literal strings — do NOT use template literals or
// dynamic construction like `bg-${status}`. Tailwind's content scanner requires
// literal strings to include them in the purged bundle.

export type Status = "idle" | "connecting" | "live" | "success" | "warning" | "danger";

export interface StatusStyle {
  dot: string;    // bg-* for the indicator dot
  text: string;   // text-* for label text color
  bg: string;     // bg-* for the chip/banner background
  border: string; // border-* for the chip/banner border
}

export const STATUS: Record<Status, StatusStyle> = {
  idle: {
    dot: "bg-muted",
    text: "text-subtle",
    bg: "bg-panel-2",
    border: "border-line",
  },
  connecting: {
    dot: "bg-warning",
    text: "text-warning",
    bg: "bg-warning/10",
    border: "border-warning/30",
  },
  live: {
    dot: "bg-success",
    text: "text-success",
    bg: "bg-success/10",
    border: "border-success/30",
  },
  success: {
    dot: "bg-success",
    text: "text-success",
    bg: "bg-success/10",
    border: "border-success/30",
  },
  warning: {
    dot: "bg-warning",
    text: "text-warning",
    bg: "bg-warning/10",
    border: "border-warning/30",
  },
  danger: {
    dot: "bg-danger",
    text: "text-danger",
    bg: "bg-danger/10",
    border: "border-danger/30",
  },
};
