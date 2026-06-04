"use client";

import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

// recharts does NOT auto-theme — we pass CSS var values directly as inline style strings.
// oklch(var(--token)) works in modern browsers and in SVG fill attributes (Chrome 111+, Firefox 113+, Safari 15.4+).
const chartColors = {
  line:    "oklch(var(--brand))",
  tick:    "oklch(var(--muted))",
  tooltip: {
    bg:      "oklch(var(--panel))",
    border:  "oklch(var(--border))",
    text:    "oklch(var(--text))",
    label:   "oklch(var(--subtle))",
  },
};

export function ConfidenceChart({ data }: { data: number[] }) {
  const points = data.map((v, i) => ({ i, v }));
  const latest = points.at(-1)?.v ?? 0;
  const average = points.length ? points.reduce((sum, p) => sum + p.v, 0) / points.length : 0;

  if (!points.length) {
    return (
      <div className="flex min-h-40 items-center justify-center rounded-card border border-line bg-panel-2 p-5 text-center">
        <div>
          <p className="font-bold text-text">ยังไม่มีกระแสข้อมูลทำนาย</p>
          <p className="mt-1 text-sm text-subtle">เริ่มกล้องแล้วค่าความมั่นใจจะไหลเข้ามาในกราฟนี้</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Summary pills with mono numerals */}
      <div className="flex flex-wrap gap-2 text-xs text-subtle">
        <span className="rounded-full border border-line bg-panel-2 px-3 py-1">
          ล่าสุด{" "}
          <strong className="ml-1 font-mono font-medium text-text">
            {(latest * 100).toFixed(0)}%
          </strong>
        </span>
        <span className="rounded-full border border-line bg-panel-2 px-3 py-1">
          เฉลี่ย{" "}
          <strong className="ml-1 font-mono font-medium text-text">
            {(average * 100).toFixed(0)}%
          </strong>
        </span>
        <span className="rounded-full border border-line bg-panel-2 px-3 py-1">
          จุดข้อมูล{" "}
          <strong className="ml-1 font-mono font-medium text-text">{points.length}</strong>
        </span>
      </div>

      {/* Chart — border only, no paired wide shadow */}
      <div className="h-40 w-full rounded-card border border-line bg-panel p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: -22 }}>
            <XAxis dataKey="i" hide />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: chartColors.tick }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              formatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              labelFormatter={() => "ความมั่นใจ"}
              contentStyle={{
                borderRadius: 10,
                borderColor: chartColors.tooltip.border,
                backgroundColor: chartColors.tooltip.bg,
                fontSize: 12,
                color: chartColors.tooltip.text,
              }}
              labelStyle={{ color: chartColors.tooltip.label }}
              itemStyle={{ color: chartColors.tooltip.text }}
            />
            <Line
              type="monotone"
              dataKey="v"
              stroke={chartColors.line}
              strokeWidth={2.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
