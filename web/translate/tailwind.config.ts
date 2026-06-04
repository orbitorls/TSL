import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // All semantic colors reference OKLCH CSS vars.
        // CSS var stores "L C H" channels; Tailwind wraps in oklch(... / <alpha-value>)
        // so bg-brand/95, ring-brand/20, text-brand, etc. all work.
        brand: {
          DEFAULT: "oklch(var(--brand) / <alpha-value>)",
          strong: "oklch(var(--brand-strong) / <alpha-value>)",
          soft: "oklch(var(--brand-soft) / <alpha-value>)",
          fg: "oklch(var(--brand-fg) / <alpha-value>)",
        },
        page: "oklch(var(--page) / <alpha-value>)",
        panel: "oklch(var(--panel) / <alpha-value>)",
        "panel-2": "oklch(var(--panel-2) / <alpha-value>)",
        border: "oklch(var(--border) / <alpha-value>)",
        line: "oklch(var(--line) / <alpha-value>)",
        subtle: "oklch(var(--subtle) / <alpha-value>)",
        muted: "oklch(var(--muted) / <alpha-value>)",
        text: "oklch(var(--text) / <alpha-value>)",
        ink: "oklch(var(--ink) / <alpha-value>)",
        success: "oklch(var(--success) / <alpha-value>)",
        warning: "oklch(var(--warning) / <alpha-value>)",
        danger: "oklch(var(--danger) / <alpha-value>)",
      },
      borderRadius: {
        // Named scale to avoid over-rounding (replaces rounded-3xl / rounded-[2rem])
        panel: "16px",   // app shell, top-level panels
        card: "14px",    // inner cards, overlays
        field: "10px",   // inputs, selects, small buttons
      },
      boxShadow: {
        // Pick border OR shadow per element — never both with blur >16px
        card: "0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
        panel: "0 4px 14px rgba(0,0,0,0.07)",
        popover: "0 8px 30px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.08)",
      },
      fontFamily: {
        // Single Thai family with weight contrast; IBM Plex Mono for numeric data only
        thai: ["var(--font-noto-thai)", "Sarabun", "Prompt", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "Menlo", "Consolas", "monospace"],
      },
      letterSpacing: {
        // Use sparingly — only on one eyebrow / Latin-only labels
        operator: "0.08em",
      },
    },
  },
  plugins: [],
};

export default config;
