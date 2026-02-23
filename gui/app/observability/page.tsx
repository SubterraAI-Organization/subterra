"use client";

import React, { useEffect, useMemo, useState } from "react";
import { ProgressBar } from "../_components/ProgressBar";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";
const GENO_BASE = process.env.NEXT_PUBLIC_GENOTYPE_BASE_URL ?? "http://localhost:8002";

type SystemMetrics = {
  counts: Record<string, number>;
  annotation_times_utc: string[];
  model_version_times_utc: string[];
  model_final_loss: Array<number | null>;
  model_num_samples: Array<number | null>;
  qc_rejected_deltas_px: number[];
  mask_nonzero_fraction: number[];
  trait_total_root_length: number[];
};

type GenoStats = { samples: number; markers: number; values: number };

function fmtInt(n: unknown) {
  if (typeof n !== "number" || !Number.isFinite(n)) return "—";
  return String(Math.trunc(n));
}

export default function Page() {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [system, setSystem] = useState<SystemMetrics | null>(null);
  const [genoStats, setGenoStats] = useState<GenoStats | null>(null);
  const [plotNonce, setPlotNonce] = useState(() => Date.now());
  const [runId, setRunId] = useState<string>("");

  const q = useMemo(() => {
    const params = new URLSearchParams();
    if (runId.trim()) params.set("run_id", runId.trim());
    params.set("ts", String(plotNonce));
    return params.toString();
  }, [plotNonce, runId]);

  const plotUrl = `${API_BASE}/system/figure1-inset.png?${q}`;
  const apiZipUrl = `${API_BASE}/observability/export.zip${runId.trim() ? `?run_id=${encodeURIComponent(runId.trim())}` : ""}`;
  const genoZipUrl = `${GENO_BASE}/observability/export.zip`;

  async function refreshAll() {
    setError(null);
    setLoading(true);
    try {
      const [sys, gstats] = await Promise.all([
        fetch(`${API_BASE}/system/metrics${runId.trim() ? `?run_id=${encodeURIComponent(runId.trim())}` : ""}`).then((r) =>
          r.ok ? (r.json() as Promise<SystemMetrics>) : null
        ),
        fetch(`${GENO_BASE}/stats`).then((r) => (r.ok ? (r.json() as Promise<GenoStats>) : null))
      ]);
      setSystem(sys);
      setGenoStats(gstats);
      setPlotNonce(Date.now());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refreshAll();
    const id = setInterval(refreshAll, 15000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const cards = useMemo(() => {
    const c = system?.counts ?? {};
    return [
      { title: "Images phenotyped", value: fmtInt(c.images_phenotyped) },
      { title: "Corrected pairs", value: fmtInt(c.corrected_pairs) },
      { title: "Model versions", value: fmtInt(c.model_versions) },
      { title: "QC rejections", value: fmtInt(c.qc_rejections) }
    ];
  }, [system]);

  return (
    <main style={{ display: "grid", gap: 16 }}>
      <header className="panel">
        <h1 style={{ margin: 0, fontSize: 18 }}>Observability</h1>
        <p style={{ margin: "6px 0 0", color: "var(--muted)", fontSize: 13 }}>
          System-level metrics to support the paper (end-to-end counts, QC behavior, versioned training, and trait output
          distributions).
        </p>
        <div className="hint" style={{ marginTop: 8 }}>
          API: {API_BASE} · Genotype: {GENO_BASE}
        </div>
        {error ? <div style={{ marginTop: 10, color: "var(--danger)", fontSize: 13 }}>{error}</div> : null}
      </header>

      <section className="panel">
        <div className="panelTitle">Quick Counts</div>
        <ProgressBar active={loading} label={loading ? "Refreshing…" : undefined} reserveSpace />
        <div className="row" style={{ marginTop: 10, alignItems: "center", gap: 10 }}>
          <div className="field" style={{ minWidth: 280 }}>
            <div className="label">Run / Study ID (optional)</div>
            <input
              className="input"
              value={runId}
              placeholder="e.g., CS2026_R01"
              onChange={(e) => setRunId(e.target.value)}
            />
            <div className="hint">When set, downloads and plots are filtered to this run_id.</div>
          </div>
          <button className="btn" onClick={() => setRunId("")} disabled={loading || !runId.trim()}>
            Clear
          </button>
        </div>
        <div className="cards" style={{ marginTop: 10 }}>
          {cards.map((c) => (
            <div key={c.title} className="panel">
              <div className="panelTitle">{c.title}</div>
              <div style={{ fontSize: 22 }}>{c.value}</div>
            </div>
          ))}
          <div className="panel">
            <div className="panelTitle">Genotype rows</div>
            <div style={{ fontSize: 22 }}>{genoStats ? fmtInt(genoStats.values) : "—"}</div>
            <div className="hint">samples {genoStats ? fmtInt(genoStats.samples) : "—"} · markers {genoStats ? fmtInt(genoStats.markers) : "—"}</div>
          </div>
        </div>

        <div className="row" style={{ marginTop: 12, alignItems: "center", gap: 10 }}>
          <button className="btn" onClick={() => void refreshAll()} disabled={loading}>
            Refresh
          </button>
          <a className="btn" href={apiZipUrl} download="subterra_observability_export.zip">
            Download all (ZIP)
          </a>
          <a className="btn" href={genoZipUrl} download="genotype_observability_export.zip">
            Download genotype (ZIP)
          </a>
          <a
            className="btn"
            href={`${API_BASE}/observability/analyses.csv${runId.trim() ? `?run_id=${encodeURIComponent(runId.trim())}` : ""}`}
            download="analyses.csv"
          >
            Analyses CSV
          </a>
          <a
            className="btn"
            href={`${API_BASE}/observability/annotations.csv${runId.trim() ? `?run_id=${encodeURIComponent(runId.trim())}` : ""}`}
            download="annotations.csv"
          >
            Annotations CSV
          </a>
          <a
            className="btn"
            href={`${API_BASE}/observability/model_versions.csv${runId.trim() ? `?run_id=${encodeURIComponent(runId.trim())}` : ""}`}
            download="model_versions.csv"
          >
            Model versions CSV
          </a>
          <a
            className="btn"
            href={`${API_BASE}/observability/train_jobs.csv${runId.trim() ? `?run_id=${encodeURIComponent(runId.trim())}` : ""}`}
            download="train_jobs.csv"
          >
            Train jobs CSV
          </a>
          <a
            className="btn"
            href={`${API_BASE}/observability/qc_rejections.csv${runId.trim() ? `?run_id=${encodeURIComponent(runId.trim())}` : ""}`}
            download="qc_rejections.csv"
          >
            QC rejects CSV
          </a>
        </div>

        <div className="row" style={{ marginTop: 10, alignItems: "center", gap: 10 }}>
          <a className="btn" href={plotUrl} download="figure1_inset_panels.png">
            Download inset PNG
          </a>
          <a
            className="btn"
            href={`${API_BASE}/system/metrics${runId.trim() ? `?run_id=${encodeURIComponent(runId.trim())}` : ""}`}
            download="system_metrics.json"
          >
            System metrics JSON
          </a>
          <div className="hint">Inset plot: {API_BASE}/system/figure1-inset.png</div>
        </div>

        <div className="row" style={{ marginTop: 10, alignItems: "center", gap: 10 }}>
          <a className="btn" href={`${GENO_BASE}/observability/mapping_runs.csv`} download="mapping_runs.csv">
            Mapping runs CSV
          </a>
          <a className="btn" href={`${GENO_BASE}/observability/mapping_hits.csv`} download="mapping_hits.csv">
            Mapping hits CSV (latest run)
          </a>
          <a className="btn" href={`${GENO_BASE}/observability/markers.csv`} download="markers.csv">
            Markers CSV
          </a>
          <a className="btn" href={`${GENO_BASE}/observability/samples.csv`} download="samples.csv">
            Samples CSV
          </a>
        </div>
      </section>

      <section className="panel">
        <div className="panelTitle">Inset Plot (paper-style)</div>
        <div className="hint" style={{ marginTop: 6 }}>
          Generated from stored artifacts: analyses table (phenotyping runs), annotation `meta.json` (corrections + mask
          coverage), model registry (versions/loss), and QC rejection logs.
        </div>
        <div style={{ marginTop: 12, overflowX: "auto" }}>
          <img src={plotUrl} alt="Figure 1 inset system metrics" style={{ width: "100%", maxWidth: 1200, height: "auto" }} />
        </div>
      </section>

      <section className="panel">
        <div className="panelTitle">What to run to populate metrics</div>
        <ul style={{ margin: "10px 0 0", paddingLeft: 18, color: "var(--muted)", fontSize: 13, lineHeight: 1.5 }}>
          <li>Phenotype: run batch analysis so the API writes to the <code>analyses</code> table.</li>
          <li>Correct: save corrected masks so annotation <code>meta.json</code> is created.</li>
          <li>Train: run retraining so new model versions and training metrics are registered.</li>
          <li>QC rejections: try saving a mismatched-size mask once; it will be rejected and logged.</li>
          <li>Genotype mapping: upload markers CSV in Genotyping; stats show up here automatically.</li>
        </ul>
      </section>

      <section className="panel">
        <div className="panelTitle">Raw JSON (system metrics)</div>
        <pre
          style={{
            marginTop: 10,
            padding: 12,
            borderRadius: 12,
            border: "1px solid var(--border)",
            background: "rgba(0,0,0,0.20)",
            overflow: "auto",
            fontSize: 12
          }}
        >
          {system ? JSON.stringify(system, null, 2) : "No data yet."}
        </pre>
      </section>
    </main>
  );
}
