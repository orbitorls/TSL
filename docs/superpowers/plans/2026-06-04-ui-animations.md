# UI Animation Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tasteful, polished motion to the TSL Translate web UI so that predictions, status changes, and step transitions feel fluid and alive rather than instant.

**Architecture:** Pure CSS keyframes added to `globals.css` (gated behind `prefers-reduced-motion: no-preference`), applied via Tailwind utility classes and `key` props to re-trigger entrance animations on content change. No animation library is needed — the existing OKLCH design system already has `fade-rise` and `pulse-live` as the pattern to follow. Recharts line animation is re-enabled with a short duration.

**Tech Stack:** Tailwind CSS v3, CSS `@keyframes`, Next.js 14 app router, Recharts

---

## File Map

| File | Change |
|------|--------|
| `web/translate/app/globals.css` | Add 3 new keyframes + utility classes |
| `web/translate/components/CameraStage.tsx` | Apply animation classes to prediction label, badges, overlays |
| `web/translate/components/Stepper.tsx` | Staggered step entrance, animated underline, animated checkmark |
| `web/translate/components/TranscriptPanel.tsx` | Pending badge entrance, copy button bounce |
| `web/translate/components/ConfidenceChart.tsx` | Re-enable recharts line animation |

---

### Task 1: Add New Keyframes to globals.css

**Files:**
- Modify: `web/translate/app/globals.css:135-168`

- [ ] **Step 1: Add the three new keyframes and utility classes inside the existing `@media (prefers-reduced-motion: no-preference)` block**

Replace the current keyframes block (lines 135–160) with:

```css
@media (prefers-reduced-motion: no-preference) {
  @keyframes fade-rise {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @keyframes pulse-live {
    0%, 100% { box-shadow: 0 0 0 0 oklch(var(--success) / 0.45); }
    50%       { box-shadow: 0 0 0 5px oklch(var(--success) / 0); }
  }

  /* Badge/pill slides down from above the top bar */
  @keyframes slide-down {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* Prediction label or card pops in with a slight scale */
  @keyframes pop-in {
    from { opacity: 0; transform: translateX(-50%) scale(0.88); }
    to   { opacity: 1; transform: translateX(-50%) scale(1); }
  }

  /* Active underline in Stepper grows from left */
  @keyframes scale-x-in {
    from { transform: scaleX(0); transform-origin: left; }
    to   { transform: scaleX(1); transform-origin: left; }
  }

  .animate-fade-rise {
    animation: fade-rise 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
  }

  .animate-pulse-live {
    animation: pulse-live 2.2s ease-in-out infinite;
  }

  .animate-slide-down {
    animation: slide-down 0.3s cubic-bezier(0.16, 1, 0.3, 1) both;
  }

  .animate-pop-in {
    animation: pop-in 0.35s cubic-bezier(0.16, 1, 0.3, 1) both;
  }

  .animate-scale-x-in {
    animation: scale-x-in 0.3s cubic-bezier(0.16, 1, 0.3, 1) both;
  }
}

/* Reduced-motion safety net — prevents any inadvertent animation */
@media (prefers-reduced-motion: reduce) {
  .animate-fade-rise,
  .animate-pulse-live,
  .animate-slide-down,
  .animate-pop-in,
  .animate-scale-x-in {
    animation: none;
  }
}
```

- [ ] **Step 2: Verify the CSS is syntactically valid**

```powershell
cd D:\TSL\web\translate
npx next build --no-lint 2>&1 | Select-String -Pattern "error|Error" | Select-Object -First 10
```

Expected: No CSS parse errors. (Build may fail for other reasons — only CSS errors matter here.)

- [ ] **Step 3: Commit**

```powershell
cd D:\TSL
git add web/translate/app/globals.css
git commit -m "feat(ui): add pop-in, slide-down, scale-x-in keyframes for animation polish"
```

---

### Task 2: Animate CameraStage Prediction Label and Badges

**Files:**
- Modify: `web/translate/components/CameraStage.tsx`

The key trick: add `key={overlayLabel}` to the prediction label div so React unmounts/remounts it on every new prediction — this re-triggers the entrance animation. Same trick for status badge using `key={live ? 'live' : streaming ? 'connecting' : 'idle'}`.

**Important:** The `pop-in` keyframe uses `translateX(-50%)` to work alongside the existing `-translate-x-1/2` class (the `translate` transform from Tailwind and the keyframe must agree on X offset, otherwise the label jumps). The keyframe already incorporates the `translateX(-50%)` so the element lands correctly.

- [ ] **Step 1: Update the prediction label overlay div**

Find this block (around line 172):
```tsx
      {/* Prediction label overlay — solid, no glass */}
      {hasPrediction && (
        <div className="absolute bottom-4 left-1/2 z-10 w-[min(92%,28rem)] -translate-x-1/2 rounded-card bg-brand px-5 py-3 text-center shadow-panel">
          <span className="text-[clamp(2rem,6vw,3rem)] font-bold leading-tight text-brand-fg">
            {overlayLabel}
          </span>
        </div>
      )}
```

Replace with:
```tsx
      {/* Prediction label overlay — solid, no glass */}
      {hasPrediction && (
        <div
          key={overlayLabel}
          className="animate-pop-in absolute bottom-4 left-1/2 z-10 w-[min(92%,28rem)] rounded-card bg-brand px-5 py-3 text-center shadow-panel"
        >
          <span className="text-[clamp(2rem,6vw,3rem)] font-bold leading-tight text-brand-fg">
            {overlayLabel}
          </span>
        </div>
      )}
```

Note: `-translate-x-1/2` is removed from className because the `pop-in` keyframe already handles `translateX(-50%)` in both `from` and `to` states. Without this, the Tailwind class and keyframe transform fight each other and the label drifts on entrance.

- [ ] **Step 2: Add slide-down to the status badge (left side) with key for re-animation**

Find this block (around line 125):
```tsx
        <span
          className={`rounded-full border px-3 py-1 text-xs font-bold backdrop-blur ${
            live
              ? "border-success/40 bg-black/55 text-success"
              : streaming
                ? "border-warning/40 bg-black/55 text-warning"
                : "border-white/15 bg-black/40 text-white/70"
          }`}
        >
          {live ? "LIVE · พร้อมทำนาย" : streaming ? "กำลังเชื่อมต่อกล้อง" : "รอเปิดกล้อง"}
        </span>
```

Replace with:
```tsx
        <span
          key={live ? "live" : streaming ? "connecting" : "idle"}
          className={`animate-slide-down rounded-full border px-3 py-1 text-xs font-bold backdrop-blur ${
            live
              ? "border-success/40 bg-black/55 text-success"
              : streaming
                ? "border-warning/40 bg-black/55 text-warning"
                : "border-white/15 bg-black/40 text-white/70"
          }`}
        >
          {live ? "LIVE · พร้อมทำนาย" : streaming ? "กำลังเชื่อมต่อกล้อง" : "รอเปิดกล้อง"}
        </span>
```

- [ ] **Step 3: Add slide-down to the conditional right-side badges**

Find the `handBadge && (...)` span and the `noHandsDetected && (...)` span and the `waitingLandmarks && (...)` span (around lines 139–162). Add `animate-slide-down` to each:

```tsx
          {handBadge && (
            <span
              className={`animate-slide-down rounded-full border px-3 py-1 text-xs font-semibold backdrop-blur ${
                handsDetected.left || handsDetected.right
                  ? "border-success/35 bg-black/55 text-success"
                  : "border-warning/35 bg-black/55 text-warning"
              }`}
            >
              {handBadge}
            </span>
          )}
          {noHandsDetected && (
            <span className="animate-slide-down rounded-full border border-warning/40 bg-black/60 px-3 py-1 text-[11px] font-medium leading-snug text-warning backdrop-blur">
              Server ไม่เห็นมือ — ถอยห่างให้เห็นไหล่+มือในเฟรม
            </span>
          )}
          {waitingLandmarks && (
            <span className="animate-slide-down rounded-full border border-white/20 bg-black/50 px-3 py-1 text-xs font-semibold text-white/85 backdrop-blur">
              รอ skeleton…
            </span>
          )}
          {predictionStatus === "no_hand" && live && (
            <span className="animate-slide-down rounded-full border border-danger/40 bg-black/60 px-3 py-1 text-xs font-semibold text-danger backdrop-blur">
              ไม่พบมือ (server)
            </span>
          )}
```

- [ ] **Step 4: Add fade-rise to the idle overlay card**

Find this block (around line 197):
```tsx
      {!streaming && (
        <div className="absolute inset-0 flex items-center justify-center bg-page/92 p-6 text-center backdrop-blur-sm">
          <div className="max-w-sm rounded-card border border-line bg-panel px-6 py-5 shadow-card">
```

Replace the inner card div:
```tsx
      {!streaming && (
        <div className="absolute inset-0 flex items-center justify-center bg-page/92 p-6 text-center backdrop-blur-sm">
          <div className="animate-fade-rise max-w-sm rounded-card border border-line bg-panel px-6 py-5 shadow-card">
```

- [ ] **Step 5: Add fade-rise to the connecting overlay**

Find this block (around line 208):
```tsx
      {streaming && !live && (
        <div className="absolute inset-x-4 bottom-4 z-10 rounded-card border border-warning/30 bg-warning/10 px-4 py-3 text-sm text-warning backdrop-blur">
```

Replace with:
```tsx
      {streaming && !live && (
        <div className="animate-fade-rise absolute inset-x-4 bottom-4 z-10 rounded-card border border-warning/30 bg-warning/10 px-4 py-3 text-sm text-warning backdrop-blur">
```

- [ ] **Step 6: Start dev server and visually verify**

```powershell
cd D:\TSL\web\translate
npm run dev
```

Open `http://localhost:3000`. Verify:
- Status badge slides down smoothly when camera state changes (connect camera, then disconnect)
- Prediction label pops in with scale on each new sign — stays centered, does NOT jump or drift horizontally
- Idle overlay card fades up on load
- Connecting overlay fades in when camera connects before WS is live

- [ ] **Step 7: Commit**

```powershell
cd D:\TSL
git add web/translate/components/CameraStage.tsx
git commit -m "feat(ui): animate camera stage badges and prediction label"
```

---

### Task 3: Animate Stepper — Staggered Entrance, Animated Underline

**Files:**
- Modify: `web/translate/components/Stepper.tsx`

Stagger: each step card gets an `animationDelay` of `index * 60ms`. This is applied as an inline style since Tailwind's arbitrary delay (`[animation-delay:120ms]`) requires the full value as a Tailwind class which doesn't support runtime values. The checkmark SVG gets a short stroke-dasharray animation via CSS class.

- [ ] **Step 1: Replace the Stepper implementation entirely**

Replace all of `web/translate/components/Stepper.tsx` with:

```tsx
const steps = [
  { n: 1, title: "ตั้งค่าระบบ", desc: "เลือกแทร็กและชุดโมเดล" },
  { n: 2, title: "โหลดโมเดล", desc: "ตรวจสอบสถานะพร้อมใช้" },
  { n: 3, title: "เปิดกล้อง", desc: "เริ่มแปลแบบสด" },
  { n: 4, title: "ดูผลแปล", desc: "ข้อความไทยและความมั่นใจ" },
];

export function Stepper({ active }: { active: number }) {
  return (
    <nav
      aria-label="ลำดับการใช้งาน"
      className="grid grid-cols-2 gap-2 sm:grid-cols-2 lg:grid-cols-4"
    >
      {steps.map((s, i) => {
        const isDone = active > s.n;
        const isActive = active === s.n;

        return (
          <div
            key={s.n}
            className={`animate-fade-rise relative overflow-hidden rounded-card border px-4 py-3 text-sm transition-colors ${
              isActive
                ? "border-brand bg-panel shadow-card ring-2 ring-brand/15"
                : isDone
                  ? "border-success/30 bg-success/5"
                  : "border-line bg-panel text-subtle"
            }`}
            style={{ animationDelay: `${i * 60}ms` }}
            aria-current={isActive ? "step" : undefined}
          >
            <div className="flex items-start gap-3">
              {/* Step number badge */}
              <span
                className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-bold transition-colors ${
                  isActive
                    ? "bg-brand text-brand-fg"
                    : isDone
                      ? "bg-success text-panel"
                      : "bg-panel-2 text-muted"
                }`}
              >
                {isDone ? (
                  <svg
                    key="check"
                    width="12"
                    height="12"
                    viewBox="0 0 12 12"
                    fill="none"
                    aria-hidden
                    className="animate-draw-check"
                  >
                    <path
                      d="M2 6l3 3 5-5"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeDasharray="12"
                      strokeDashoffset="0"
                    />
                  </svg>
                ) : (
                  s.n
                )}
              </span>

              <div className="min-w-0">
                <strong
                  className={`block text-sm font-semibold transition-colors ${
                    isActive ? "text-brand" : isDone ? "text-success" : "text-subtle"
                  }`}
                >
                  {s.title}
                </strong>
                <p className="mt-0.5 text-xs leading-5 text-muted">{s.desc}</p>
              </div>
            </div>

            {/* Active underline — grows from left */}
            {isActive && (
              <span
                className="animate-scale-x-in absolute inset-x-0 bottom-0 h-0.5 bg-brand"
                aria-hidden
              />
            )}
          </div>
        );
      })}
    </nav>
  );
}
```

- [ ] **Step 2: Add the `draw-check` keyframe to globals.css**

Inside the `@media (prefers-reduced-motion: no-preference)` block, add after `scale-x-in`:

```css
  /* SVG checkmark stroke draws in from left to right */
  @keyframes draw-check {
    from { stroke-dashoffset: 12; }
    to   { stroke-dashoffset: 0; }
  }

  .animate-draw-check path {
    stroke-dasharray: 12;
    stroke-dashoffset: 12;
    animation: draw-check 0.35s cubic-bezier(0.16, 1, 0.3, 1) 0.1s both;
  }
```

Also add `animate-draw-check` to the reduced-motion reset block:
```css
@media (prefers-reduced-motion: reduce) {
  .animate-fade-rise,
  .animate-pulse-live,
  .animate-slide-down,
  .animate-pop-in,
  .animate-scale-x-in,
  .animate-draw-check path {
    animation: none;
    stroke-dashoffset: 0;
  }
}
```

- [ ] **Step 3: Visually verify in browser**

Open `http://localhost:3000`. On page load:
- All 4 Stepper cards should fade up with a staggered delay (card 1 first, card 4 last, ~60ms between each)
- The active underline should slide in from left to right
- When a step completes (active moves to next), the checkmark should draw itself in

- [ ] **Step 4: Commit**

```powershell
cd D:\TSL
git add web/translate/components/Stepper.tsx web/translate/app/globals.css
git commit -m "feat(ui): stagger stepper entrance, animate active underline and checkmark"
```

---

### Task 4: Animate TranscriptPanel — Pending Badge and Copy Button

**Files:**
- Modify: `web/translate/components/TranscriptPanel.tsx`

- [ ] **Step 1: Add pop-in to the pending badge and scale-pulse to copy button**

Replace the entire `TranscriptPanel.tsx` with:

```tsx
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
          key={chipLabel}
          className={`animate-slide-down shrink-0 rounded-full border px-3 py-1 text-xs font-semibold ${chipStatus.border} ${chipStatus.bg} ${chipStatus.text}`}
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
          <div className="animate-pop-in mt-4 rounded-card border border-brand/20 bg-brand-soft px-3 py-2 text-sm text-subtle">
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
          className={`rounded-field px-4 py-2 text-sm font-semibold shadow-card transition-all disabled:cursor-not-allowed disabled:opacity-40 ${
            copied
              ? "scale-105 bg-success text-panel"
              : "bg-brand text-brand-fg hover:bg-brand-strong"
          }`}
        >
          {copied ? "คัดลอกแล้ว ✓" : "คัดลอก"}
        </button>
      </div>
    </div>
  );
}
```

Key changes:
- Header chip: `key={chipLabel}` + `animate-slide-down` to re-animate when status changes
- Pending badge: `animate-pop-in` class (the `pop-in` keyframe uses `translateX(-50%)` — but this element is NOT absolutely positioned, so we need a different version)

**Wait — the pending badge is `position: static` (in normal flow), not `position: absolute`. The `pop-in` keyframe uses `translateX(-50%)` which would shift it sideways when applied to a flow element.**

Fix: add a second keyframe `pop-in-static` to globals.css for non-positioned elements. After Step 1, proceed to Step 2.

- [ ] **Step 2: Add `pop-in-static` keyframe to globals.css**

Inside the `@media (prefers-reduced-motion: no-preference)` block, add after `pop-in`:

```css
  /* pop-in for flow elements (not absolutely positioned) */
  @keyframes pop-in-static {
    from { opacity: 0; transform: scale(0.88); }
    to   { opacity: 1; transform: scale(1); }
  }

  .animate-pop-in-static {
    animation: pop-in-static 0.35s cubic-bezier(0.16, 1, 0.3, 1) both;
  }
```

Also add `animate-pop-in-static` to the reduced-motion block:
```css
  .animate-pop-in-static {
    animation: none;
  }
```

- [ ] **Step 3: Fix pending badge class in TranscriptPanel.tsx**

Change `animate-pop-in` to `animate-pop-in-static` on the pending badge div:

```tsx
        {hasPending && (
          <div className="animate-pop-in-static mt-4 rounded-card border border-brand/20 bg-brand-soft px-3 py-2 text-sm text-subtle">
```

- [ ] **Step 4: Visually verify in browser**

With the dev server running at `http://localhost:3000`:
- Status chip should slide down when it changes (idle → กำลังทาย → มีข้อความ)
- Pending badge should pop in each time a new sign is being predicted
- Copy button should briefly scale up (`scale-105`) when clicked, then snap back to normal as it returns to brand color

- [ ] **Step 5: Commit**

```powershell
cd D:\TSL
git add web/translate/components/TranscriptPanel.tsx web/translate/app/globals.css
git commit -m "feat(ui): animate transcript panel status chip, pending badge, copy button"
```

---

### Task 5: Re-enable Recharts Line Animation

**Files:**
- Modify: `web/translate/components/ConfidenceChart.tsx:80-88`

The `isAnimationActive={false}` flag was set to avoid the recharts default 1500ms animation, which feels sluggish for streaming data. We re-enable it with a short 400ms duration so new data points flow in smoothly without lag.

- [ ] **Step 1: Update the Line component props**

Find this block (around line 80):
```tsx
            <Line
              type="monotone"
              dataKey="v"
              stroke={chartColors.line}
              strokeWidth={2.5}
              dot={false}
              isAnimationActive={false}
            />
```

Replace with:
```tsx
            <Line
              type="monotone"
              dataKey="v"
              stroke={chartColors.line}
              strokeWidth={2.5}
              dot={false}
              isAnimationActive={true}
              animationDuration={400}
              animationEasing="ease-out"
            />
```

- [ ] **Step 2: Visually verify in browser**

Start the camera and perform a sign. Verify that as new confidence points arrive, the line smoothly extends rather than jumping. If the animation feels jittery (because data arrives faster than 400ms), reduce `animationDuration` to `200`.

- [ ] **Step 3: Commit**

```powershell
cd D:\TSL
git add web/translate/components/ConfidenceChart.tsx
git commit -m "feat(ui): re-enable recharts confidence line animation at 400ms"
```

---

## Self-Review

**Spec coverage check:**
- Prediction label entrance animation → Task 2 Step 1 (pop-in with key reset per label)
- Camera badges slide in → Task 2 Steps 2–3 (slide-down with key resets)
- Idle/connecting overlays → Task 2 Steps 4–5 (fade-rise)
- Stepper stagger entrance → Task 3 Step 1 (animationDelay on each card)
- Stepper active underline → Task 3 Step 1 (scale-x-in)
- Stepper checkmark draw → Task 3 Steps 1–2 (draw-check keyframe)
- Transcript status chip re-animates on change → Task 4 Step 1 (key + slide-down)
- Pending badge pops in → Task 4 Steps 1–3 (pop-in-static)
- Copy button success bounce → Task 4 Step 1 (scale-105 on copied state)
- Confidence chart line animates → Task 5

**Placeholder scan:** No TBD, TODO, or vague steps found. All code is complete.

**Type consistency:** No new types introduced. `animationDelay` is a valid inline style string property in React. `isAnimationActive`, `animationDuration`, `animationEasing` are valid Recharts `Line` props.

**Known constraint:** The `pop-in` keyframe encodes `translateX(-50%)` and is only safe on `position: absolute` elements centered with `left-1/2`. For flow elements, `pop-in-static` (no X offset) must be used instead. This distinction is documented in the plan and applied correctly.
