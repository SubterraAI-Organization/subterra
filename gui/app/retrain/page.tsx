"use client";

import React, { useEffect, useState } from "react";
import { ProgressBar } from "../_components/ProgressBar";

type TrainJob = {
  job_id: string;
  status: string;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
  planned_version_id?: string | null;
  produced_version_id?: string | null;
  log: string[];
};

type ModelVersion = { id: string; created_at: string; metrics?: any; train_config?: any };
type ModelsResponse = { unet_current?: string | null; unet_versions?: ModelVersion[] };

type AnnotationStats = {
  total_annotations: number;
  by_camera_model: Record<string, number>;
  missing_camera_model: number;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";

export default function Page() {
  const [runId, setRunId] = useState<string>("");
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(2);
  const [lr, setLr] = useState(1e-4);
  const [imageSize, setImageSize] = useState(512);
  const [cameraModel, setCameraModel] = useState<string>("");
  const [onlyCorrected, setOnlyCorrected] = useState<boolean>(false);
  const [preserveAspectRatio, setPreserveAspectRatio] = useState<boolean>(true);

  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [jobs, setJobs] = useState<TrainJob[]>([]);
  const [stats, setStats] = useState<AnnotationStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const latestJob = jobs[0] ?? null;

  useEffect(() => {
    const saved = localStorage.getItem("subterra_run_id");
    if (saved) setRunId(saved);
  }, []);

  useEffect(() => {
    localStorage.setItem("subterra_run_id", runId);
  }, [runId]);

  async function refresh() {
    setError(null);
    try {
      const [m, j, s] = await Promise.all([
        fetch(`${API_BASE_URL}/models`).then((r) => (r.ok ? r.json() : null)),
        fetch(`${API_BASE_URL}/train/jobs`).then((r) => (r.ok ? r.json() : [])),
        fetch(`${API_BASE_URL}/annotations/stats`).then((r) => (r.ok ? r.json() : null))
      ]);
      setModels(m);
      setJobs(j);
      setStats(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  useEffect(() => {
    void refresh();
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, []);

  async function startTraining() {
    setStarting(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/train/unet`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          epochs,
          batch_size: batchSize,
          lr,
          image_size: imageSize,
          camera_model: cameraModel || null,
          only_corrected: onlyCorrected,
          preserve_aspect_ratio: preserveAspectRatio,
          run_id: runId.trim() || null
        })
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `HTTP ${res.status}`);
      }
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  }

  return (
    <main style={{ display: "grid", gap: 16 }}>
      <header className="panel">
        <h1 style={{ margin: 0, fontSize: 18 }}>Re-training (U-Net)</h1>
        <p style={{ margin: "6px 0 0", color: "var(--muted)", fontSize: 13 }}>
          Fine-tune the current U-Net using saved annotation pairs in `data/annotations/`. Each run creates a new version.
        </p>
        <div className="hint" style={{ marginTop: 8 }}>
          API: {API_BASE_URL}
        </div>
        <div className="row" style={{ marginTop: 10, alignItems: "flex-end" }}>
          <div className="field">
            <div className="label">Run / Study ID</div>
            <input
              className="input"
              value={runId}
              placeholder="e.g., CS_round13_14"
              onChange={(e) => setRunId(e.target.value)}
            />
            <div className="hint">Tag the training run so model versions show up under this `run_id` in exports.</div>
          </div>
        </div>
        <div style={{ marginTop: 10 }}>
          <ProgressBar active={starting || latestJob?.status === "running"} label={latestJob?.status === "running" ? "Training…" : starting ? "Starting…" : undefined} />
        </div>
      </header>

      {error ? <div className="panel" style={{ color: "var(--danger)" }}>{error}</div> : null}

      <section className="grid">
        <div className="panel">
          <div className="panelTitle">Train Config</div>
          <div className="row">
            <div className="field">
              <div className="label">Epochs</div>
              <input className="input" type="number" min={1} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
            </div>
            <div className="field">
              <div className="label">Batch Size</div>
              <input className="input" type="number" min={1} value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} />
            </div>
            <div className="field">
              <div className="label">Learning Rate</div>
              <input className="input" type="number" step={1e-5} value={lr} onChange={(e) => setLr(Number(e.target.value))} />
            </div>
            <div className="field">
              <div className="label">Image Size</div>
              <input className="input" type="number" min={128} step={64} value={imageSize} onChange={(e) => setImageSize(Number(e.target.value))} />
            </div>
            <div className="field">
              <div className="label">Camera Filter</div>
              <select className="select" value={cameraModel} onChange={(e) => setCameraModel(e.target.value)}>
                <option value="">All cameras</option>
                <option value="CID CI-600 In-Situ Root Imager">CID CI-600 In-Situ Root Imager</option>
                {Object.keys(stats?.by_camera_model ?? {})
                  .sort()
                  .filter((cam) => cam !== "CID CI-600 In-Situ Root Imager")
                  .map((cam) => (
                    <option key={cam} value={cam}>
                      {cam} ({stats?.by_camera_model?.[cam] ?? 0})
                    </option>
                  ))}
              </select>
            </div>
            <div className="field">
              <div className="label">Use Only Corrected</div>
              <select className="select" value={onlyCorrected ? "yes" : "no"} onChange={(e) => setOnlyCorrected(e.target.value === "yes")}>
                <option value="no">No (use all)</option>
                <option value="yes">Yes (recommended)</option>
              </select>
            </div>
            <div className="field">
              <div className="label">Aspect Ratio</div>
              <select className="select" value={preserveAspectRatio ? "preserve" : "stretch"} onChange={(e) => setPreserveAspectRatio(e.target.value === "preserve")}>
                <option value="preserve">Preserve + pad (recommended)</option>
                <option value="stretch">Stretch to square</option>
              </select>
            </div>
            <button className="btn btnPrimary" onClick={() => void startTraining()} disabled={starting || (latestJob?.status === "running")}>
              {latestJob?.status === "running" ? "Training Running…" : "Start Training"}
            </button>
          </div>
          <div className="hint" style={{ marginTop: 10 }}>
            Dataset: {stats ? `${stats.total_annotations} annotations` : "—"} · missing camera metadata:{" "}
            {stats ? stats.missing_camera_model : "—"}
          </div>
          <div className="hint" style={{ marginTop: 6 }}>
            Tip: for CI-600 training, annotate via `Mode=Minirhizotron` in `/annotate` so camera metadata is stored.
          </div>
          <div className="hint" style={{ marginTop: 10 }}>
            Tip: annotate a few samples first in the Annotation page, then retrain here.
          </div>
        </div>

        <div className="panel">
          <div className="panelTitle">Model Versions</div>
          <div className="hint">Current: {models?.unet_current ?? "unavailable"}</div>
          <div className="list" style={{ marginTop: 10 }}>
            {(models?.unet_versions ?? []).slice().reverse().slice(0, 10).map((v) => (
              <div key={v.id} className="item" style={{ cursor: "default", gridTemplateColumns: "1fr" }}>
                <div className="meta">
                  <div className="filename">{v.id}</div>
                  <div className="status">{v.created_at}</div>
                </div>
              </div>
            ))}
            {(models?.unet_versions ?? []).length === 0 ? <div className="hint">No versions yet. First run will create `unet_v0001`.</div> : null}
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panelTitle">Latest Training Job</div>
        {!latestJob ? (
          <div className="hint">No jobs yet.</div>
        ) : (
          <>
            <div className="row">
              <div className="hint">Status: {latestJob.status}</div>
              <div className="hint">Planned: {latestJob.planned_version_id ?? "—"}</div>
              <div className="hint">Produced: {latestJob.produced_version_id ?? "—"}</div>
              {latestJob.error ? <div className="hint" style={{ color: "var(--danger)" }}>Error: {latestJob.error}</div> : null}
            </div>
            <div className="panel" style={{ marginTop: 12, padding: 12, maxHeight: 240, overflow: "auto" }}>
              <div className="hint" style={{ whiteSpace: "pre-wrap" }}>
                {(latestJob.log ?? []).join("\n")}
              </div>
            </div>
          </>
        )}
      </section>
    </main>
  );
}
