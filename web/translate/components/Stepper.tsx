const steps = [
  { n: 1, title: "ตั้งค่าระบบ", desc: "เลือกแทร็กและชุดโมเดล" },
  { n: 2, title: "โหลดโมเดล", desc: "ตรวจสอบสถานะพร้อมใช้" },
  { n: 3, title: "เปิดกล้อง", desc: "เริ่มแปลแบบสด" },
  { n: 4, title: "ดูผลแปล", desc: "ข้อความไทยและความมั่นใจ" },
];

export function Stepper({ active }: { active: number }) {
  return (
    <nav aria-label="ลำดับการใช้งาน" className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
      {steps.map((s) => {
        const isDone = active > s.n;
        const isActive = active === s.n;

        return (
          <div
            key={s.n}
            className={`relative overflow-hidden rounded-2xl border px-4 py-3 text-sm transition ${
              isActive
                ? "border-brand bg-white shadow-sm ring-4 ring-brand/10"
                : isDone
                  ? "border-brand/35 bg-white/90"
                  : "border-border bg-white/55 text-subtle"
            }`}
            aria-current={isActive ? "step" : undefined}
          >
            <div className="flex items-start gap-3">
              <span
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-bold ${
                  isActive || isDone ? "bg-brand text-white" : "bg-page text-subtle"
                }`}
              >
                {s.n}
              </span>
              <div className="min-w-0">
                <strong className={isActive || isDone ? "text-brand" : "text-subtle"}>{s.title}</strong>
                <p className="mt-0.5 text-xs leading-5 text-subtle">{s.desc}</p>
              </div>
            </div>
            {isActive && <span className="absolute inset-x-0 bottom-0 h-1 bg-brand-light" aria-hidden />}
          </div>
        );
      })}
    </nav>
  );
}
