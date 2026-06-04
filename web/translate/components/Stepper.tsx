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
      {steps.map((s) => {
        const isDone = active > s.n;
        const isActive = active === s.n;

        return (
          <div
            key={s.n}
            className={`relative overflow-hidden rounded-card border px-4 py-3 text-sm transition-colors ${
              isActive
                ? "border-brand bg-panel shadow-card ring-2 ring-brand/15"
                : isDone
                  ? "border-success/30 bg-success/5"
                  : "border-line bg-panel text-subtle"
            }`}
            aria-current={isActive ? "step" : undefined}
          >
            <div className="flex items-start gap-3">
              {/* Step number badge */}
              <span
                className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
                  isActive
                    ? "bg-brand text-brand-fg"
                    : isDone
                      ? "bg-success text-panel"
                      : "bg-panel-2 text-muted"
                }`}
              >
                {isDone ? (
                  /* Checkmark for done steps */
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
                    <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : (
                  s.n
                )}
              </span>

              <div className="min-w-0">
                <strong
                  className={`block text-sm font-semibold ${
                    isActive ? "text-brand" : isDone ? "text-success" : "text-subtle"
                  }`}
                >
                  {s.title}
                </strong>
                <p className="mt-0.5 text-xs leading-5 text-muted">{s.desc}</p>
              </div>
            </div>

            {/* Active underline */}
            {isActive && (
              <span
                className="absolute inset-x-0 bottom-0 h-0.5 bg-brand"
                aria-hidden
              />
            )}
          </div>
        );
      })}
    </nav>
  );
}
