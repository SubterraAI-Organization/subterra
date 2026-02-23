"use client";

import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ProgressBar } from "./_components/ProgressBar";

type Health = { status: string; models_loaded: string[]; device: string };
type Dashboard = {
  counts: { annotations: number; analyses: number; model_versions: number; train_jobs: number; api_keys: number };
  current_unet_version?: string | null;
  recent_annotations: Array<{ annotation_id: string; created_at: string; original_filename: string }>;
  recent_analyses: Array<{
    id: number;
    filename: string;
    created_at: string;
    model_type: string;
    model_version: string;
    root_count: number;
    total_root_length: number;
  }>;
  recent_train_jobs: Array<{
    job_id: string;
    status: string;
    created_at: string;
    planned_version_id: string;
    produced_version_id: string;
  }>;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";

export default function Page() {
  const [health, setHealth] = useState<Health | null>(null);
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const lastHashRef = useRef<string>("");
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load({ first }: { first: boolean }) {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      if (first) setInitialLoading(true);

      let showSpinnerTimeout: ReturnType<typeof setTimeout> | null = null;
      if (!first) {
        showSpinnerTimeout = setTimeout(() => {
          if (!cancelled) setRefreshing(true);
        }, 350);
      }

      try {
        const [h, r] = await Promise.all([
          fetch(`${API_BASE_URL}/health`, { signal: controller.signal }).then((res) => res.json()),
          fetch(`${API_BASE_URL}/dashboard`, { signal: controller.signal }).then(async (res) => {
            if (!res.ok) throw new Error(await res.text());
            return res.json();
          })
        ]);
        if (cancelled) return;

        const nextHash = JSON.stringify({ h, r });
        if (nextHash !== lastHashRef.current) {
          lastHashRef.current = nextHash;
          setHealth(h);
          setDashboard(r);
        }
        setError(null);
        setLastUpdated(new Date().toISOString());
      } catch (e) {
        if (!cancelled && !(e instanceof DOMException && e.name === "AbortError")) {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (showSpinnerTimeout) clearTimeout(showSpinnerTimeout);
        if (!cancelled) {
          setInitialLoading(false);
          setRefreshing(false);
        }
      }
    }

    void load({ first: true });
    const id = setInterval(() => void load({ first: false }), 5000);
    return () => {
      cancelled = true;
      abortRef.current?.abort();
      clearInterval(id);
    };
  }, []);

  return (
    <main style={{ display: "grid", gap: 16 }}>
      <header className="panel">
        <h1 style={{ margin: 0, fontSize: 18 }}>Dashboard</h1>
        <p style={{ margin: "6px 0 0", color: "var(--muted)", fontSize: 13 }}>
          Human-in-the-loop root segmentation: annotate → retrain → phenotype → repeat.
        </p>
        <div className="hint" style={{ marginTop: 8 }}>
          API: {API_BASE_URL}
        </div>
        <div style={{ marginTop: 10 }}>
          <ProgressBar
            active={initialLoading || refreshing}
            label={initialLoading ? "Loading…" : refreshing ? "Refreshing…" : undefined}
            reserveSpace
          />
        </div>
        <div className="hint" style={{ marginTop: 6 }}>
          Last updated: {lastUpdated ?? "—"}
        </div>
        {error ? <div style={{ marginTop: 10, color: "var(--danger)", fontSize: 13 }}>{error}</div> : null}
      </header>

      <section className="panel">
        <div className="panelTitle">Quick Actions</div>
        <div className="row">
          <Link className="btn btnPrimary" href="/annotate">
            Start Annotation
          </Link>
          <Link className="btn" href="/retrain">
            Re-train U-Net
          </Link>
          <Link className="btn" href="/phenotype">
            Run Phenotyping
          </Link>
        </div>
        <div className="hint" style={{ marginTop: 10 }}>
          Suggested loop: annotate → save corrected masks → retrain (creates a new model version) → phenotype.
        </div>
      </section>

      <section className="cards">
        <div className="panel">
          <div className="panelTitle">Annotations</div>
          <div style={{ fontSize: 22 }}>{dashboard?.counts.annotations ?? "—"}</div>
          <div className="hint">Saved corrected mask pairs</div>
        </div>
        <div className="panel">
          <div className="panelTitle">Analyses</div>
          <div style={{ fontSize: 22 }}>{dashboard?.counts.analyses ?? "—"}</div>
          <div className="hint">Inference runs stored in DB</div>
        </div>
        <div className="panel">
          <div className="panelTitle">U-Net Versions</div>
          <div style={{ fontSize: 22 }}>{dashboard?.counts.model_versions ?? "—"}</div>
          <div className="hint">Current: {dashboard?.current_unet_version ?? "—"}</div>
        </div>
        <div className="panel">
          <div className="panelTitle">Training Jobs</div>
          <div style={{ fontSize: 22 }}>{dashboard?.counts.train_jobs ?? "—"}</div>
          <div className="hint">Most recent status below</div>
        </div>
        <div className="panel">
          <div className="panelTitle">API Keys</div>
          <div style={{ fontSize: 22 }}>{dashboard?.counts.api_keys ?? "—"}</div>
          <div className="hint">
            For ingestion · <Link href="/api" style={{ textDecoration: "underline" }}>manage</Link>
          </div>
        </div>
      </section>

      <section className="grid">
        <div className="panel">
          <div className="panelTitle">Recent Analyses</div>
          {dashboard?.recent_analyses?.length ? (
            <div style={{ overflow: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    {["filename", "root_count", "total_root_length", "model", "created_at"].map((h) => (
                      <th
                        key={h}
                        style={{
                          textAlign: "left",
                          padding: "8px 6px",
                          color: "var(--muted)",
                          borderBottom: "1px solid var(--border)"
                        }}
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dashboard.recent_analyses.map((a) => (
                    <tr key={a.id}>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>{a.filename}</td>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>{a.root_count}</td>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>{a.total_root_length}</td>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>
                        {a.model_type} {a.model_version ? `(${a.model_version})` : ""}
                      </td>
                      <td style={{ padding: "8px 6px", borderBottom: "1px solid var(--border)" }}>{a.created_at}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="hint">No analyses yet. Run Phenotyping or Annotation to generate records.</div>
          )}
        </div>

        <div className="panel">
          <div className="panelTitle">System</div>
          <div className="hint">API health: {health ? health.status : "…"}</div>
          <div className="hint">Device: {health ? health.device : "…"}</div>
          <div className="hint">Loaded models: {health ? health.models_loaded.join(", ") || "none" : "…"}</div>
          <div className="panelTitle" style={{ marginTop: 12 }}>
            Recent Annotations
          </div>
          {dashboard?.recent_annotations?.length ? (
            <div className="list">
              {dashboard.recent_annotations.slice(0, 6).map((a) => (
                <div key={a.annotation_id} className="item" style={{ cursor: "default", gridTemplateColumns: "1fr" }}>
                  <div className="meta">
                    <div className="filename">{a.original_filename || a.annotation_id}</div>
                    <div className="status">{a.created_at}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="hint">No saved annotations yet.</div>
          )}

          <div className="panelTitle" style={{ marginTop: 12 }}>
            Recent Training Jobs
          </div>
          {dashboard?.recent_train_jobs?.length ? (
            <div className="list">
              {dashboard.recent_train_jobs.slice(0, 6).map((j) => (
                <div key={j.job_id} className="item" style={{ cursor: "default", gridTemplateColumns: "1fr" }}>
                  <div className="meta">
                    <div className="filename">{j.status}</div>
                    <div className="status">
                      {j.produced_version_id ? `produced ${j.produced_version_id}` : `planned ${j.planned_version_id}`} ·{" "}
                      {j.created_at}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="hint">No training jobs yet.</div>
          )}
        </div>
      </section>
    </main>
  );
}
