const steps = [
  { n: 1, title: "ตั้งค่าระบบ", desc: "เลือกแทร็กและชุดโมเดล" },
  { n: 2, title: "โหลดโมเดล", desc: "ตรวจสอบสถานะพร้อมใช้" },
  { n: 3, title: "เปิดกล้อง", desc: "เริ่มแปลแบบสด" },
  { n: 4, title: "ดูผลแปล", desc: "ข้อความไทยและความมั่นใจ" },
];

export function Stepper({ active }: { active: number }) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      {steps.map((s) => (
        <div
          key={s.n}
          className={`rounded-xl border px-3 py-2 text-sm ${
            active >= s.n ? "border-brand bg-white shadow-sm" : "border-border bg-white/60 text-subtle"
          }`}
        >
          <strong className="text-brand">{s.n}) {s.title}</strong>
          <p className="mt-0.5 text-xs text-subtle">{s.desc}</p>
        </div>
      ))}
    </div>
  );
}
