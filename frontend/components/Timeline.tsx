"use client";

type TimelineStep = {
  step: number;
  title: string;
  detail: string;
};

export default function Timeline({ steps }: { steps: TimelineStep[] }) {
  return (
    <div className="space-y-2">
      {steps.map((s) => (
        <div key={s.step} className="flex gap-3">
          <div className="mt-1 h-2 w-2 rounded-full bg-emerald-400" />
          <div>
            <div className="text-sm font-semibold">
              {s.step}. {s.title}
            </div>
            <div className="text-xs text-slate-400">{s.detail}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
