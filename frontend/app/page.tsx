"use client";

import { useEffect, useMemo, useState } from "react";
import Timeline from "../components/Timeline";
import Overlay from "../components/Overlay";

type Health = {
  ok: boolean;
  model_path: string;
  model_file_exists: boolean;
  model_loaded: boolean;
  providers_used: string[] | null;
  load_error: string | null;
};

type Detection = {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox_xyxy: [number, number, number, number];
};

type TimelineStep = {
  step: number;
  title: string;
  detail: string;
};

type InferResponse = {
  detections: Detection[];
  unique_objects: string[];
  target_object: string | null;
  master_key: string;
  timeline: TimelineStep[];
  latency_ms: number;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Page() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthErr, setHealthErr] = useState("");

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");

  const [loading, setLoading] = useState(false);
  const [inferErr, setInferErr] = useState("");

  const [result, setResult] = useState<InferResponse | null>(null);

  // --- fetch health on load ---
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/health`);
        const j = await r.json();
        setHealth(j);
      } catch (e: any) {
        setHealthErr(String(e));
      }
    })();
  }, []);

  // --- preview URL ---
  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onPickFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInferErr("");
    setResult(null);
    const f = e.target.files?.[0];
    if (f) setFile(f);
  };

  const onInfer = async () => {
    if (!file) return;
    setLoading(true);
    setInferErr("");
    setResult(null);

    try {
      const fd = new FormData();
      fd.append("file", file);

      const r = await fetch(`${API_BASE}/infer`, {
        method: "POST",
        body: fd,
      });

      if (!r.ok) {
        const txt = await r.text();
        throw new Error(txt || `Infer failed: ${r.status}`);
      }

      const j: InferResponse = await r.json();
      setResult(j);
    } catch (e: any) {
      setInferErr(String(e));
    } finally {
      setLoading(false);
    }
  };

  const copyKey = async () => {
    if (!result?.master_key) return;
    try {
      await navigator.clipboard.writeText(result.master_key);
      alert("Master key copied!");
    } catch {
      alert("Couldn’t copy automatically. Please copy manually.");
    }
  };

  const detectionsByClass = useMemo(() => {
    const map: Record<string, number> = {};
    result?.detections?.forEach((d) => {
      map[d.class_name] = (map[d.class_name] || 0) + 1;
    });
    return map;
  }, [result]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="mx-auto max-w-6xl px-4 pt-6">
        <div className="flex items-center justify-between gap-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-xl bg-slate-800 text-xl">
              🔐
            </div>
            <div>
              <h1 className="text-lg font-semibold">
                EnCryptic Vision-Entropy Demo
              </h1>
              <p className="text-xs text-slate-400">
                Car-cam frames → objects → master seed
              </p>
            </div>
          </div>

          {/* Health */}
          <div className="flex items-center gap-2 text-xs text-slate-300">
            <span
              className={`h-2 w-2 rounded-full ${
                health?.model_loaded ? "bg-emerald-400" : "bg-rose-400"
              }`}
            />
            {healthErr && (
              <span className="text-rose-400">API down: {healthErr}</span>
            )}
            {!healthErr && !health && <span>Checking backend…</span>}
            {health && (
              <div className="space-y-0.5">
                <div>
                  Backend:{" "}
                  <span className="font-medium text-slate-100">
                    {API_BASE}
                  </span>
                </div>
                <div>
                  Model loaded:{" "}
                  <span
                    className={`font-semibold ${
                      health.model_loaded ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {String(health.model_loaded)}
                  </span>
                </div>
                <div className="text-[11px] text-slate-400">
                  Provider: {health.providers_used?.join(", ") || "—"}
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main grid */}
      <main className="mx-auto grid max-w-6xl grid-cols-1 gap-4 px-4 py-6 md:grid-cols-2">
        {/* Left: Upload */}
        <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
          <h2 className="mb-3 text-sm font-semibold">1) Upload a frame</h2>

          <label className="block cursor-pointer rounded-xl border border-dashed border-slate-700 bg-slate-950/40 p-4 hover:bg-slate-950/60">
            <input
              type="file"
              accept="image/*"
              onChange={onPickFile}
              className="hidden"
            />
            <div className="flex items-center gap-3">
              <div className="text-xl">📷</div>
              <div>
                {!file ? (
                  <>
                    <div className="text-sm font-medium">
                      Click to choose an image
                    </div>
                    <div className="text-xs text-slate-400">
                      Road / carcam / urban frames work best
                    </div>
                  </>
                ) : (
                  <>
                    <div className="text-sm font-medium">{file.name}</div>
                    <div className="text-xs text-slate-400">
                      {(file.size / 1024).toFixed(1)} KB
                    </div>
                  </>
                )}
              </div>
            </div>
          </label>

          {previewUrl && (
            <div className="mt-3 overflow-hidden rounded-xl border border-slate-800 bg-black">
              <img
                src={previewUrl}
                alt="preview"
                className="max-h-[380px] w-full object-contain"
              />
            </div>
          )}

          <button
            onClick={onInfer}
            disabled={!file || loading}
            className="mt-3 w-full rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-4 py-2 text-sm font-semibold disabled:opacity-60"
          >
            {loading ? "Running inference…" : "Generate master key"}
          </button>

          {inferErr && (
            <div className="mt-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-200">
              {inferErr}
            </div>
          )}

          {result?.latency_ms != null && (
            <div className="mt-3 inline-block rounded-lg border border-slate-800 bg-slate-950/40 px-2 py-1 text-xs text-slate-300">
              Latency: {result.latency_ms} ms
            </div>
          )}
        </section>

        {/* Right: Results */}
        <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
          <h2 className="mb-3 text-sm font-semibold">2) Results</h2>

          {!result && (
            <div className="rounded-xl border border-dashed border-slate-700 bg-slate-950/40 p-4 text-sm text-slate-400">
              Upload an image and run inference to see detections + timeline.
            </div>
          )}

          {result && (
            <>
              {/* Key box */}
              <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-xs text-slate-400">
                      Master key (SHA-256)
                    </div>
                    <div className="text-[11px] text-slate-500">
                      Stable for the same visual scene
                    </div>
                  </div>
                  <button
                    onClick={copyKey}
                    className="rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-xs font-semibold hover:bg-slate-800"
                  >
                    Copy
                  </button>
                </div>
                <code className="mt-2 block break-all text-xs leading-relaxed text-slate-100">
                  {result.master_key}
                </code>
              </div>

              {/* Unique + target */}
              <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                  <div className="mb-1 text-xs text-slate-400">
                    Unique objects
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {result.unique_objects.length === 0 && (
                      <span className="text-xs text-slate-500">
                        None detected
                      </span>
                    )}
                    {result.unique_objects.map((u) => (
                      <span
                        key={u}
                        className="rounded-full border border-slate-700 bg-slate-900 px-2 py-0.5 text-xs"
                      >
                        {u}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                  <div className="mb-1 text-xs text-slate-400">
                    Random target
                  </div>
                  {result.target_object ? (
                    <span className="rounded-full border border-violet-500/60 bg-violet-500/10 px-2 py-0.5 text-xs font-semibold text-violet-200">
                      {result.target_object}
                    </span>
                  ) : (
                    <span className="text-xs text-slate-500">No target</span>
                  )}
                </div>
              </div>

              {/* Counts */}
              <div className="mt-3">
                <div className="mb-2 text-xs text-slate-400">Detections</div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {Object.keys(detectionsByClass).length === 0 && (
                    <span className="text-xs text-slate-500">No detections</span>
                  )}
                  {Object.entries(detectionsByClass).map(([cls, n]) => (
                    <div
                      key={cls}
                      className="grid place-items-center rounded-xl border border-slate-800 bg-slate-950/40 p-3 text-center"
                    >
                      <div className="text-xl font-extrabold">{n}</div>
                      <div className="text-xs text-slate-400">{cls}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Timeline */}
              <div className="mt-3 rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                <div className="mb-2 text-xs text-slate-400">
                  Realtime pipeline
                </div>
                <Timeline steps={result.timeline} />
              </div>

              {/* Optional overlay over image */}
              {previewUrl && result.detections?.length > 0 && (
                <div className="mt-3">
                  <Overlay imageUrl={previewUrl} detections={result.detections} />
                </div>
              )}

              {/* raw json */}
              <details className="mt-3">
                <summary className="cursor-pointer text-xs text-slate-400">
                  Raw detections JSON
                </summary>
                <pre className="mt-2 whitespace-pre-wrap rounded-xl border border-slate-800 bg-black/40 p-3 text-xs">
                  {JSON.stringify(result.detections, null, 2)}
                </pre>
              </details>
            </>
          )}
        </section>
      </main>

      <footer className="mx-auto max-w-6xl px-4 pb-6 text-center text-xs text-slate-500">
        EnCryptic Demo • Vision-AI entropy → cryptographic seed
      </footer>
    </div>
  );
}
