"use client";

import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function ConfidenceChart({ data }: { data: number[] }) {
  const points = data.map((v, i) => ({ i, v }));

  if (!points.length) {
    return <p className="text-sm text-subtle">ยังไม่มีกระแสข้อมูลทำนาย</p>;
  }

  return (
    <div className="h-40 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points}>
          <XAxis dataKey="i" hide />
          <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
          <Tooltip formatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
          <Line type="monotone" dataKey="v" stroke="#13a08f" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
