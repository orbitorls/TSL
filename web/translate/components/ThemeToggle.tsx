"use client";

import { useEffect, useState } from "react";

/** Reads the current theme from the DOM (set by the no-flash inline script). */
function getIsDark(): boolean {
  return document.documentElement.classList.contains("dark");
}

/**
 * Light/dark theme toggle.
 * - Initialises after mount (avoids SSR hydration mismatch).
 * - Reads current theme from documentElement.classList (set by the no-flash script).
 * - Persists choice in localStorage as "tsl-theme".
 * - Respects prefers-reduced-motion for the icon crossfade.
 */
export function ThemeToggle() {
  const [isDark, setIsDark] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    setIsDark(getIsDark());
  }, []);

  const toggle = () => {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("tsl-theme", next ? "dark" : "light");
    } catch {
      // localStorage unavailable — no-op
    }
  };

  // Render a placeholder with the same dimensions before mount to prevent layout shift
  if (!mounted) {
    return (
      <div
        className="h-9 w-9 rounded-full border border-line bg-panel-2"
        aria-hidden
      />
    );
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={isDark ? "สลับเป็นธีมสว่าง" : "สลับเป็นธีมมืด"}
      aria-pressed={isDark}
      title={isDark ? "Light mode" : "Dark mode"}
      className="flex h-9 w-9 items-center justify-center rounded-full border border-line bg-panel text-muted transition hover:bg-panel-2 hover:text-text"
    >
      {isDark ? (
        /* Sun icon — indicates "switch to light" */
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
        >
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
        </svg>
      ) : (
        /* Moon icon — indicates "switch to dark" */
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
        >
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      )}
    </button>
  );
}
