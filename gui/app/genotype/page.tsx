"use client";

import React, { useEffect, useMemo, useState } from "react";
import { ProgressBar } from "../_components/ProgressBar";
import { PaginatedTable, type PaginatedTableColumn } from "../_components/PaginatedTable";

const GENO_BASE = process.env.NEXT_PUBLIC_GENOTYPE_BASE_URL ?? "http://localhost:8002";
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";

type Stats = { samples: number; markers: number; values: number };
type UploadResp = { samples_upserted: number; markers_upserted: number; values_upserted: number };
type MappingRow = {
  marker_name: string;
  n: number;
  effect?: number | null;
  p_value?: number | null;
  p_adjusted?: number | null;
  r2?: number | null;
  lod?: number | null;
};
type MappingRunResp = { phenotype_field: string; method: string; p_adjust: string; rows: MappingRow[] };
type MappingRunQuery = {
  phenotype_field: string;
  method: string;
  p_adjust: string;
  allow_filename_fallback: boolean;
  max_markers: number;
  min_n: number;
};
type HistoryItem = { created_at: string; phenotype_field: string; marker_name: string; n: number; effect: number };
type MarkerEffectClass = {
  genotype_class: string;
  n: number;
  mean: number;
  median: number;
  q1: number;
  q3: number;
  whisker_low: number;
  whisker_high: number;
  min: number;
  max: number;
};
type MarkerEffectResp = {
  marker_name: string;
  phenotype_field: string;
  n_samples: number;
  classes: MarkerEffectClass[];
};
type MarkerFile = {
  name: string;
  source_dir?: string;
  format?: string;
  bytes: number;
  modified_at: string;
  uploadable: boolean;
  ingestable?: boolean;
  ingest_note?: string;
};

function fmtR(r: number) {
  if (!Number.isFinite(r)) return "—";
  return r.toFixed(4);
}

function fmtP(p: number | null | undefined) {
  if (p == null || !Number.isFinite(p)) return "—";
  if (p === 0) return "0";
  if (p < 1e-4) return p.toExponential(2);
  return p.toFixed(6);
}

function effectMetricLabel(method: string) {
  return method === "pearson" ? "r" : method === "anova" || method === "lod" ? "F" : "beta";
}

function negLog10(p: number | null | undefined) {
  if (p == null || !Number.isFinite(p) || p <= 0) return 0;
  return -Math.log10(Math.max(1e-300, p));
}

function markerKey(f: Pick<MarkerFile, "name" | "source_dir">) {
  return `${f.source_dir ?? ""}::${f.name}`;
}

function markerUrl(file: MarkerFile | null): string {
  if (!file) return "#";
  const q = new URLSearchParams({ name: file.name });
  if (file.source_dir) q.set("source_dir", file.source_dir);
  return `${GENO_BASE}/markers/download?${q.toString()}`;
}

export default function Page() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [markerFiles, setMarkerFiles] = useState<MarkerFile[]>([]);
  const [markerPick, setMarkerPick] = useState<string>("");

  const [uploading, setUploading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResp | null>(null);
  const [mapUploading, setMapUploading] = useState(false);
  const [mapResult, setMapResult] = useState<{ rows_seen: number; rows_upserted: number; rows_skipped: number } | null>(null);
  const [replaceExisting, setReplaceExisting] = useState<boolean>(true);
  const [ingestEvery, setIngestEvery] = useState<number>(200);
  const [ingestMaxMarkers, setIngestMaxMarkers] = useState<number>(5000);

  const [phenotypeField, setPhenotypeField] = useState<string>("total_root_length");
  const [method, setMethod] = useState<string>("linear");
  const [pAdjust, setPAdjust] = useState<string>("bh");
  const [mappingMaxMarkers, setMappingMaxMarkers] = useState<number>(2000);
  const [mappingMinN, setMappingMinN] = useState<number>(6);
  const [allowFilenameFallback, setAllowFilenameFallback] = useState<boolean>(false);
  const [running, setRunning] = useState(false);
  const [runResult, setRunResult] = useState<MappingRunResp | null>(null);
  const [lastRunQuery, setLastRunQuery] = useState<MappingRunQuery | null>(null);
  const [showAllHits, setShowAllHits] = useState(false);
  const [plotNonce, setPlotNonce] = useState<number>(() => Date.now());
  const [markerEffect, setMarkerEffect] = useState<MarkerEffectResp | null>(null);
  const [markerEffectLoading, setMarkerEffectLoading] = useState(false);
  const [markerEffectError, setMarkerEffectError] = useState<string | null>(null);
  const [plotSvgMarkup, setPlotSvgMarkup] = useState<string>("");
  const [plotLoading, setPlotLoading] = useState(false);
  const [plotError, setPlotError] = useState<string | null>(null);

  const selectedMarker = useMemo(
    () => markerFiles.find((f) => markerKey(f) === markerPick) ?? markerFiles[0] ?? null,
    [markerFiles, markerPick]
  );

  async function refresh() {
    try {
      const [s, h, mf] = await Promise.all([
        fetch(`${GENO_BASE}/stats`).then((r) => (r.ok ? r.json() : null)),
        fetch(`${GENO_BASE}/mapping/history?limit=25`).then((r) => (r.ok ? r.json() : [])),
        fetch(`${GENO_BASE}/markers/list`).then((r) => (r.ok ? r.json() : { files: [] }))
      ]);
      setStats(s);
      setHistory(h);

      const files = ((mf as { files?: MarkerFile[] })?.files ?? []).filter((x) => x && typeof x.name === "string");
      setMarkerFiles(files);
      if (!files.length) {
        setMarkerPick("");
        return;
      }
      if (!files.some((f) => markerKey(f) === markerPick)) {
        const preferred =
          files.find((f) => f.name === "SAP_imputed.hmp" && (f.ingestable ?? false)) ??
          files.find((f) => f.ingestable) ??
          files.find((f) => f.name === "arabidopsis_magic_chr1_subset.csv") ??
          files[0]!;
        setMarkerPick(markerKey(preferred));
      }
    } catch {
      // ignore; error shown on interactions
    }
  }

  useEffect(() => {
    void refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  async function uploadCsv(file: File) {
    setError(null);
    setUploadResult(null);
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      const q = new URLSearchParams({ replace_existing: replaceExisting ? "true" : "false" });
      const res = await fetch(`${GENO_BASE}/markers/upload?${q.toString()}`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      setUploadResult((await res.json()) as UploadResp);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  }

  async function ingestServerMarker() {
    if (!selectedMarker) return;
    setError(null);
    setUploadResult(null);
    setIngesting(true);
    try {
      const res = await fetch(`${GENO_BASE}/markers/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: selectedMarker.name,
          source_dir: selectedMarker.source_dir ?? null,
          max_markers: Math.max(1, Math.floor(ingestMaxMarkers || 1)),
          every: Math.max(1, Math.floor(ingestEvery || 1)),
          replace_existing: replaceExisting
        })
      });
      if (!res.ok) throw new Error(await res.text());
      const out = (await res.json()) as UploadResp;
      setUploadResult(out);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setIngesting(false);
    }
  }

  async function uploadSampleIdMap(file: File) {
    setError(null);
    setMapResult(null);
    setMapUploading(true);
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      const res = await fetch(`${API_BASE}/observability/sample_id_map/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      setMapResult((await res.json()) as { rows_seen: number; rows_upserted: number; rows_skipped: number });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setMapUploading(false);
    }
  }

  async function runMapping() {
    setError(null);
    setRunning(true);
    setRunResult(null);
    setLastRunQuery(null);
    setMarkerEffect(null);
    setMarkerEffectError(null);
    setShowAllHits(false);
    try {
      const req: MappingRunQuery = {
        phenotype_field: phenotypeField,
        method,
        p_adjust: pAdjust,
        allow_filename_fallback: allowFilenameFallback,
        max_markers: Math.max(100, Math.floor(mappingMaxMarkers || 100)),
        min_n: Math.max(3, Math.floor(mappingMinN || 3))
      };
      const res = await fetch(`${GENO_BASE}/mapping/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req)
      });
      if (!res.ok) throw new Error(await res.text());
      setRunResult((await res.json()) as MappingRunResp);
      setLastRunQuery(req);
      setPlotNonce(Date.now());
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  const top10 = useMemo(() => (runResult?.rows ?? []).slice(0, 10), [runResult]);
  const effectLabel = effectMetricLabel(runResult?.method ?? method);

  const mappingPlotUrl = useMemo(() => {
    if (!lastRunQuery || !runResult) return "";
    const q = new URLSearchParams({
      phenotype_field: lastRunQuery.phenotype_field,
      method: lastRunQuery.method,
      p_adjust: lastRunQuery.p_adjust,
      max_markers: String(lastRunQuery.max_markers),
      min_n: String(lastRunQuery.min_n),
      allow_filename_fallback: lastRunQuery.allow_filename_fallback ? "true" : "false",
      width: "1500",
      height: "620",
      ts: String(plotNonce)
    });
    return `${GENO_BASE}/mapping/plot.svg?${q.toString()}`;
  }, [lastRunQuery, plotNonce, runResult]);

  const signalEffect = useMemo(() => {
    const pts = (runResult?.rows ?? [])
      .filter((r) => r.effect != null && Number.isFinite(r.effect) && r.p_value != null && Number.isFinite(r.p_value) && (r.p_value as number) > 0)
      .map((r) => ({
        marker: r.marker_name,
        effect: r.effect as number,
        neglogp: negLog10(r.p_value),
        isFdrSig: r.p_adjusted != null && Number.isFinite(r.p_adjusted) && (r.p_adjusted as number) <= 0.05
      }))
      .sort((a, b) => b.neglogp - a.neglogp)
      .slice(0, 240);

    const maxY = Math.max(2, ...pts.map((p) => p.neglogp));
    const maxAbsEffect = Math.max(0.01, ...pts.map((p) => Math.abs(p.effect)));
    const labels = [...pts].sort((a, b) => b.neglogp - a.neglogp).slice(0, 4);
    return { points: pts, maxY, maxAbsEffect, labels };
  }, [runResult]);

  const runSummary = useMemo(() => {
    const rows = runResult?.rows ?? [];
    const nominal = rows.filter((r) => (r.p_value ?? 1) <= 0.05).length;
    const fdr = rows.filter((r) => (r.p_adjusted ?? 1) <= 0.05).length;
    const top = rows[0] ?? null;
    return { nRows: rows.length, nominal, fdr, top };
  }, [runResult]);

  useEffect(() => {
    if (!mappingPlotUrl) {
      setPlotSvgMarkup("");
      setPlotError(null);
      setPlotLoading(false);
      return;
    }
    let active = true;
    setPlotLoading(true);
    setPlotError(null);
    fetch(mappingPlotUrl, { cache: "no-store" })
      .then(async (res) => {
        const txt = await res.text();
        if (!res.ok) {
          throw new Error(`${res.status} ${res.statusText}`);
        }
        if (!txt.includes("<svg")) {
          throw new Error("Plot response was not SVG content");
        }
        return txt;
      })
      .then((txt) => {
        if (!active) return;
        setPlotSvgMarkup(txt);
      })
      .catch((e) => {
        if (!active) return;
        setPlotSvgMarkup("");
        setPlotError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!active) return;
        setPlotLoading(false);
      });
    return () => {
      active = false;
    };
  }, [mappingPlotUrl]);

  useEffect(() => {
    const marker = runSummary.top?.marker_name ?? "";
    if (!lastRunQuery || !runResult || !marker) {
      setMarkerEffect(null);
      setMarkerEffectError(null);
      return;
    }

    let active = true;
    setMarkerEffectLoading(true);
    setMarkerEffectError(null);
    const q = new URLSearchParams({
      phenotype_field: lastRunQuery.phenotype_field,
      marker_name: marker,
      allow_filename_fallback: lastRunQuery.allow_filename_fallback ? "true" : "false"
    });
    fetch(`${GENO_BASE}/mapping/marker-effect?${q.toString()}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return (await res.json()) as MarkerEffectResp;
      })
      .then((payload) => {
        if (!active) return;
        setMarkerEffect(payload);
      })
      .catch((e) => {
        if (!active) return;
        setMarkerEffect(null);
        setMarkerEffectError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!active) return;
        setMarkerEffectLoading(false);
      });

    return () => {
      active = false;
    };
  }, [lastRunQuery, runResult, runSummary.top?.marker_name]);

  const mappingHitColumns: Array<PaginatedTableColumn<MappingRow>> = [
    { id: "marker", header: "marker", sortValue: (r) => r.marker_name, cell: (r) => r.marker_name },
    { id: "n", header: "n", sortValue: (r) => r.n, cell: (r) => r.n },
    { id: "effect", header: effectLabel, sortValue: (r) => r.effect ?? null, cell: (r) => (r.effect == null ? "—" : fmtR(r.effect)) },
    { id: "p", header: "p", sortValue: (r) => r.p_value ?? null, cell: (r) => fmtP(r.p_value ?? null) },
    { id: "p_adj", header: "p_adj", sortValue: (r) => r.p_adjusted ?? null, cell: (r) => fmtP(r.p_adjusted ?? null) },
    { id: "r2", header: "r2", sortValue: (r) => r.r2 ?? null, cell: (r) => (r.r2 == null ? "—" : fmtR(r.r2)) },
    { id: "lod", header: "lod", sortValue: (r) => r.lod ?? null, cell: (r) => (r.lod == null ? "—" : fmtR(r.lod)) }
  ];

  const historyColumns: Array<PaginatedTableColumn<HistoryItem>> = [
    { id: "created_at", header: "created_at", sortValue: (r) => r.created_at, cell: (r) => r.created_at },
    { id: "phenotype", header: "phenotype", sortValue: (r) => r.phenotype_field, cell: (r) => r.phenotype_field },
    { id: "marker", header: "marker", sortValue: (r) => r.marker_name, cell: (r) => r.marker_name },
    { id: "n", header: "n", sortValue: (r) => r.n, cell: (r) => r.n },
    { id: "effect", header: "effect", sortValue: (r) => r.effect, cell: (r) => fmtR(r.effect) }
  ];

  return (
    <main style={{ display: "grid", gap: 16 }}>
      <header className="panel">
        <h1 style={{ margin: 0, fontSize: 18 }}>Genotyping / Mapping</h1>
        <p style={{ margin: "6px 0 0", color: "var(--muted)", fontSize: 13 }}>
          Load a real marker matrix (CSV/VCF/HapMap), then map stored phenotypes (API `analyses` table) to markers.
        </p>
        <div className="hint" style={{ marginTop: 8 }}>
          Genotype service: {GENO_BASE} · API: {API_BASE}
        </div>
        {error ? <div style={{ marginTop: 10, color: "var(--danger)", fontSize: 13 }}>{error}</div> : null}
      </header>

      <section className="cards">
        <div className="panel">
          <div className="panelTitle">Samples</div>
          <div style={{ fontSize: 22 }}>{stats?.samples ?? "—"}</div>
          <div className="hint">Unique sample_id rows</div>
        </div>
        <div className="panel">
          <div className="panelTitle">Markers</div>
          <div style={{ fontSize: 22 }}>{stats?.markers ?? "—"}</div>
          <div className="hint">Marker columns</div>
        </div>
        <div className="panel">
          <div className="panelTitle">Values</div>
          <div style={{ fontSize: 22 }}>{stats?.values ?? "—"}</div>
          <div className="hint">sample×marker entries</div>
        </div>
        <div className="panel">
          <div className="panelTitle">Phenotypes</div>
          <div style={{ fontSize: 22 }}>DB-backed</div>
          <div className="hint">From API `analyses` table</div>
        </div>
      </section>

      <section className="grid">
        <div className="panel">
          <div className="panelTitle">1) Load Marker Data</div>
          <ProgressBar active={uploading || ingesting} label={uploading ? "Uploading CSV…" : ingesting ? "Ingesting marker file…" : undefined} reserveSpace />
          <div className="hint">
            Use server-side ingest for files in `data/markers` or `genotype_data`. VCF/HapMap are supported. `.vep` files are annotation-only and cannot be used as genotype matrices.
          </div>
          <div className="row" style={{ marginTop: 10, alignItems: "center", gap: 10 }}>
            <select className="select" value={selectedMarker ? markerKey(selectedMarker) : ""} onChange={(e) => setMarkerPick(e.target.value)}>
              {markerFiles.length ? (
                markerFiles.map((f) => (
                  <option key={markerKey(f)} value={markerKey(f)}>
                    {f.name} [{f.format ?? "unknown"}]
                  </option>
                ))
              ) : (
                <option value="">No marker files found</option>
              )}
            </select>
            <a className="btn" href={markerUrl(selectedMarker)} download={selectedMarker?.name ?? "marker_file"} aria-disabled={!selectedMarker}>
              Download selected
            </a>
          </div>
          {selectedMarker ? (
            <div className="hint" style={{ marginTop: 8 }}>
              Source: {selectedMarker.source_dir ?? "—"} · format: {selectedMarker.format ?? "unknown"} · size: {selectedMarker.bytes.toLocaleString()} bytes
            </div>
          ) : null}
          {selectedMarker?.ingest_note ? (
            <div className="hint" style={{ marginTop: 8, color: "var(--danger)" }}>
              {selectedMarker.ingest_note}
            </div>
          ) : null}
          <div className="row" style={{ marginTop: 10, alignItems: "flex-end", gap: 10 }}>
            <div className="field">
              <div className="label">Marker subsample stride</div>
              <input className="input" type="number" min={1} step={1} value={ingestEvery} onChange={(e) => setIngestEvery(Number(e.target.value) || 1)} />
            </div>
            <div className="field">
              <div className="label">Max markers</div>
              <input
                className="input"
                type="number"
                min={1}
                step={1}
                value={ingestMaxMarkers}
                onChange={(e) => setIngestMaxMarkers(Number(e.target.value) || 1)}
              />
            </div>
            <div className="field">
              <div className="label">Replace existing markers</div>
              <select className="select" value={replaceExisting ? "yes" : "no"} onChange={(e) => setReplaceExisting(e.target.value === "yes")}>
                <option value="yes">Yes (clean reload)</option>
                <option value="no">No (merge/upsert)</option>
              </select>
            </div>
            <button
              className="btn btnPrimary"
              onClick={() => void ingestServerMarker()}
              disabled={!selectedMarker || !(selectedMarker.ingestable ?? false) || ingesting || uploading}
            >
              Ingest selected file
            </button>
          </div>
          <div className="hint" style={{ marginTop: 10 }}>
            Manual CSV upload is still available below (first column must be `sample_id` or `filename`).
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <input
              className="input"
              type="file"
              accept=".csv,text/csv"
              disabled={uploading || ingesting}
              onChange={(e) => {
                const f = e.currentTarget.files?.[0];
                if (!f) return;
                void uploadCsv(f);
              }}
            />
          </div>
          {uploadResult ? (
            <div className="hint" style={{ marginTop: 10 }}>
              Upserted: samples {uploadResult.samples_upserted}, markers {uploadResult.markers_upserted}, values {uploadResult.values_upserted}
            </div>
          ) : null}

          <div className="row" style={{ marginTop: 14, alignItems: "center", gap: 10 }}>
            <a className="btn" href={`${API_BASE}/observability/sample_id_map.csv`} download="sample_id_map.csv">
              Download phenotype ID map
            </a>
            <div className="hint">Map `filename` → biological `sample_id` (or tube id) before mapping.</div>
          </div>
          <ProgressBar active={mapUploading} label={mapUploading ? "Uploading ID map…" : undefined} reserveSpace />
          <div className="row" style={{ marginTop: 8 }}>
            <input
              className="input"
              type="file"
              accept=".csv,text/csv"
              disabled={mapUploading}
              onChange={(e) => {
                const f = e.currentTarget.files?.[0];
                if (!f) return;
                void uploadSampleIdMap(f);
              }}
            />
          </div>
          {mapResult ? (
            <div className="hint" style={{ marginTop: 10 }}>
              ID map updated: rows {mapResult.rows_seen} (upserted {mapResult.rows_upserted})
            </div>
          ) : null}
        </div>

        <div className="panel">
          <div className="panelTitle">2) Run Mapping</div>
          <ProgressBar active={running} label={running ? "Running mapping…" : undefined} reserveSpace />
          <div className="hint" style={{ marginTop: 6 }}>
            For quantitative traits like <code>total_root_volume</code>, prefer <code>mlm</code> or <code>farmcpu</code>. If runs are slow, set
            <code> Max markers (run)</code> to 1000-2000.
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <div className="field">
              <div className="label">Phenotype Field</div>
              <select className="select" value={phenotypeField} onChange={(e) => setPhenotypeField(e.target.value)}>
                <option value="total_root_length">total_root_length</option>
                <option value="root_count">root_count</option>
                <option value="average_root_diameter">average_root_diameter</option>
                <option value="total_root_area">total_root_area</option>
                <option value="total_root_volume">total_root_volume</option>
              </select>
            </div>
            <div className="field">
              <div className="label">Method</div>
              <select className="select" value={method} onChange={(e) => setMethod(e.target.value)}>
                <option value="linear">GWAS (Linear Regression)</option>
                <option value="glm">GWAS (GLM)</option>
                <option value="mlm">GWAS (MLM)</option>
                <option value="farmcpu">GWAS (FarmCPU)</option>
                <option value="anova">QTL (ANOVA across genotypes)</option>
                <option value="lod">QTL (LOD score)</option>
                <option value="pearson">Correlation (Pearson)</option>
              </select>
            </div>
            <div className="field">
              <div className="label">P adjust</div>
              <select className="select" value={pAdjust} onChange={(e) => setPAdjust(e.target.value)}>
                <option value="bh">FDR (BH)</option>
                <option value="bonferroni">Bonferroni</option>
                <option value="none">None</option>
              </select>
            </div>
            <div className="field">
              <div className="label">Max markers (run)</div>
              <input
                className="input"
                type="number"
                min={100}
                step={100}
                value={mappingMaxMarkers}
                onChange={(e) => setMappingMaxMarkers(Number(e.target.value) || 100)}
              />
            </div>
            <div className="field">
              <div className="label">Min overlap n</div>
              <input className="input" type="number" min={3} step={1} value={mappingMinN} onChange={(e) => setMappingMinN(Number(e.target.value) || 3)} />
            </div>
            <div className="field">
              <div className="label">Filename fallback</div>
              <select
                className="select"
                value={allowFilenameFallback ? "legacy" : "strict"}
                onChange={(e) => setAllowFilenameFallback(e.target.value === "legacy")}
              >
                <option value="strict">Strict (sample_id/tube_id only)</option>
                <option value="legacy">Legacy (allow filename aliases)</option>
              </select>
            </div>
            <button className="btn btnPrimary" onClick={() => void runMapping()} disabled={running || uploading || ingesting}>
              Run
            </button>
          </div>

          {(runResult?.rows?.length ?? 0) > 0 ? (
            <div style={{ marginTop: 12 }}>
              <div className="row" style={{ marginBottom: 10, alignItems: "center", gap: 10 }}>
                <div className="hint">
                  Latest run: {runResult?.phenotype_field} · {runResult?.method} · p_adjust {runResult?.p_adjust} · hits{" "}
                  {runResult?.rows.length ?? 0}
                </div>
                <button className="btn" style={{ padding: "6px 8px" }} onClick={() => setShowAllHits((v) => !v)}>
                  {showAllHits ? "Show top 10" : "Show all"}
                </button>
              </div>
              <PaginatedTable
                rows={showAllHits ? runResult!.rows : top10}
                columns={mappingHitColumns}
                getRowKey={(r) => r.marker_name}
                initialSort={{ columnId: "p", dir: "asc" }}
                initialPageSize={25}
                pageSizeOptions={[10, 25, 50, 100, 250]}
                searchableText={(r) => `${r.marker_name}`}
                emptyLabel="No hits."
                footerLeft={<div className="hint">{showAllHits ? "Showing all hits." : "Showing top 10 hits."}</div>}
              />

              <div style={{ marginTop: 14, display: "grid", gap: 12 }}>
                <div className="cards" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))" }}>
                  <div className="panel" style={{ padding: 10 }}>
                    <div className="panelTitle">Markers tested</div>
                    <div style={{ fontSize: 20 }}>{runSummary.nRows.toLocaleString()}</div>
                  </div>
                  <div className="panel" style={{ padding: 10 }}>
                    <div className="panelTitle">Nominal p ≤ 0.05</div>
                    <div style={{ fontSize: 20 }}>{runSummary.nominal.toLocaleString()}</div>
                  </div>
                  <div className="panel" style={{ padding: 10 }}>
                    <div className="panelTitle">FDR q ≤ 0.05</div>
                    <div style={{ fontSize: 20 }}>{runSummary.fdr.toLocaleString()}</div>
                  </div>
                  <div className="panel" style={{ padding: 10 }}>
                    <div className="panelTitle">Top marker</div>
                    <div style={{ fontSize: 12, wordBreak: "break-all" }}>{runSummary.top?.marker_name ?? "—"}</div>
                    <div className="hint" style={{ marginTop: 4 }}>
                      p {fmtP(runSummary.top?.p_value)} · {effectLabel} {runSummary.top?.effect == null ? "—" : fmtR(runSummary.top.effect)}
                    </div>
                  </div>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: 12 }}>
                  <div className="panel" style={{ padding: 10 }}>
                    <div className="panelTitle">GWAS Signal Plot (Manhattan + QQ)</div>
                    <div className="row" style={{ marginBottom: 8, gap: 8 }}>
                      <button className="btn" style={{ padding: "6px 8px" }} onClick={() => setPlotNonce(Date.now())}>
                        Refresh plot
                      </button>
                      <a className="btn" style={{ padding: "6px 8px" }} href={mappingPlotUrl} download={`gwas_${runResult?.phenotype_field}_${runResult?.method}.svg`}>
                        Download SVG
                      </a>
                    </div>
                    {plotLoading ? <div className="hint">Loading plot…</div> : null}
                    {!plotLoading && plotSvgMarkup ? (
                      <div
                        style={{
                          width: "100%",
                          borderRadius: 10,
                          border: "1px solid var(--border)",
                          background: "white",
                          minHeight: 260,
                          overflow: "auto"
                        }}
                        dangerouslySetInnerHTML={{ __html: plotSvgMarkup }}
                      />
                    ) : null}
                    {!plotLoading && !plotSvgMarkup && plotError ? (
                      <div className="hint" style={{ color: "var(--danger)" }}>
                        Plot could not be rendered: {plotError}
                      </div>
                    ) : null}
                    {!plotLoading && !plotSvgMarkup && !plotError ? <div className="hint">Run mapping to render plot.</div> : null}
                  </div>

                  <div className="panel" style={{ padding: 10 }}>
                    <div className="panelTitle">Effect vs Signal (Top markers)</div>
                    {signalEffect.points.length ? (
                      <svg viewBox="0 0 560 360" width="100%" height="auto" role="img" aria-label="Effect size versus significance plot">
                        <rect x="0" y="0" width="560" height="360" fill="rgba(0,0,0,0.18)" stroke="rgba(255,255,255,0.08)" />
                        <line x1="54" y1="18" x2="54" y2="318" stroke="rgba(255,255,255,0.2)" />
                        <line x1="54" y1="318" x2="542" y2="318" stroke="rgba(255,255,255,0.2)" />
                        <line x1="298" y1="18" x2="298" y2="318" stroke="rgba(255,255,255,0.12)" strokeDasharray="5,4" />
                        {[0.25, 0.5, 0.75, 1].map((t) => {
                          const y = 318 - t * 300;
                          return <line key={`grid-${t}`} x1="54" y1={y} x2="542" y2={y} stroke="rgba(255,255,255,0.08)" />;
                        })}
                        {signalEffect.points.map((p, i) => {
                          const x = 298 + (p.effect / signalEffect.maxAbsEffect) * 236;
                          const y = 318 - (Math.min(signalEffect.maxY, p.neglogp) / signalEffect.maxY) * 300;
                          const c = p.isFdrSig ? "#ff6b6b" : "#7aa2ff";
                          return <circle key={`${p.marker}-${i}`} cx={x} cy={y} r={2.8} fill={c} fillOpacity={0.85} />;
                        })}
                        {signalEffect.labels.map((p, i) => {
                          const x = 298 + (p.effect / signalEffect.maxAbsEffect) * 236;
                          const y = 318 - (Math.min(signalEffect.maxY, p.neglogp) / signalEffect.maxY) * 300;
                          const label = p.marker.length > 20 ? `${p.marker.slice(0, 20)}…` : p.marker;
                          return (
                            <g key={`label-${p.marker}-${i}`}>
                              <circle cx={x} cy={y} r={4.2} fill="none" stroke="#e9edff" />
                              <text x={Math.min(508, x + 6)} y={Math.max(24, y - 7 - i * 10)} fill="#e9edff" fontSize="10">
                                {label}
                              </text>
                            </g>
                          );
                        })}
                        <text x="298" y="348" textAnchor="middle" fill="#a6b0d6" fontSize="12">
                          {effectLabel} (marker effect)
                        </text>
                        <text x="18" y="172" textAnchor="middle" fill="#a6b0d6" fontSize="12" transform="rotate(-90 18 172)">
                          -log10(p)
                        </text>
                      </svg>
                    ) : (
                      <div className="hint">No plottable points (requires effect and p-value fields).</div>
                    )}
                    <div className="hint" style={{ marginTop: 6 }}>
                      Red: q ≤ 0.05. Blue: non-significant after multiple-testing correction.
                    </div>
                  </div>
                </div>

                <div className="panel" style={{ padding: 10 }}>
                  <div className="panelTitle">Top Marker Genotype-Class Boxplot</div>
                  {markerEffectLoading ? (
                    <div className="hint">Loading marker effect distribution…</div>
                  ) : markerEffect?.classes?.length ? (
                    <>
                      <div className="hint" style={{ marginBottom: 8 }}>
                        Marker: <code>{markerEffect.marker_name}</code> · n={markerEffect.n_samples}
                      </div>
                      <svg viewBox="0 0 760 360" width="100%" height="auto" role="img" aria-label="Top marker genotype class boxplot">
                        {(() => {
                          const classes = [...markerEffect.classes].sort(
                            (a, b) => Number(a.genotype_class) - Number(b.genotype_class)
                          );
                          const yMinRaw = Math.min(...classes.map((c) => c.whisker_low));
                          const yMaxRaw = Math.max(...classes.map((c) => c.whisker_high));
                          const span = Math.max(1e-9, yMaxRaw - yMinRaw);
                          const pad = span * 0.08;
                          const yMin = yMinRaw - pad;
                          const yMax = yMaxRaw + pad;
                          const x0 = 80;
                          const x1 = 730;
                          const y0 = 24;
                          const y1 = 300;
                          const step = classes.length > 1 ? (x1 - x0) / (classes.length - 1) : 0;
                          const sy = (v: number) => y1 - ((v - yMin) / Math.max(1e-9, yMax - yMin)) * (y1 - y0);
                          const sx = (i: number) => x0 + i * step;
                          const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => yMin + t * (yMax - yMin));
                          return (
                            <>
                              <rect x="0" y="0" width="760" height="360" fill="rgba(0,0,0,0.18)" stroke="rgba(255,255,255,0.08)" />
                              <line x1={x0} y1={y0} x2={x0} y2={y1} stroke="rgba(255,255,255,0.2)" />
                              <line x1={x0} y1={y1} x2={x1} y2={y1} stroke="rgba(255,255,255,0.2)" />
                              {ticks.map((v, i) => {
                                const y = sy(v);
                                return (
                                  <g key={`ytick-${i}`}>
                                    <line x1={x0} y1={y} x2={x1} y2={y} stroke="rgba(255,255,255,0.08)" />
                                    <text x={x0 - 10} y={y + 4} textAnchor="end" fill="#a6b0d6" fontSize="11">
                                      {v.toFixed(2)}
                                    </text>
                                  </g>
                                );
                              })}

                              {classes.map((c, i) => {
                                const x = sx(i);
                                const bw = Math.min(84, Math.max(48, 130 / Math.max(1, classes.length)));
                                const boxTop = sy(c.q3);
                                const boxBottom = sy(c.q1);
                                const medY = sy(c.median);
                                const meanY = sy(c.mean);
                                const whTop = sy(c.whisker_high);
                                const whBottom = sy(c.whisker_low);
                                return (
                                  <g key={`cls-${c.genotype_class}`}>
                                    <line x1={x} y1={whTop} x2={x} y2={whBottom} stroke="#a6b0d6" strokeWidth="1.5" />
                                    <line x1={x - bw * 0.28} y1={whTop} x2={x + bw * 0.28} y2={whTop} stroke="#a6b0d6" />
                                    <line x1={x - bw * 0.28} y1={whBottom} x2={x + bw * 0.28} y2={whBottom} stroke="#a6b0d6" />
                                    <rect x={x - bw / 2} y={Math.min(boxTop, boxBottom)} width={bw} height={Math.max(1, Math.abs(boxBottom - boxTop))} fill="#7aa2ff55" stroke="#7aa2ff" />
                                    <line x1={x - bw / 2} y1={medY} x2={x + bw / 2} y2={medY} stroke="#e9edff" strokeWidth="2" />
                                    <circle cx={x} cy={meanY} r={3.2} fill="#ffb454" />
                                    <text x={x} y={y1 + 18} textAnchor="middle" fill="#e9edff" fontSize="12">
                                      class {c.genotype_class}
                                    </text>
                                    <text x={x} y={y1 + 32} textAnchor="middle" fill="#a6b0d6" fontSize="11">
                                      n={c.n}
                                    </text>
                                  </g>
                                );
                              })}

                              <text x={(x0 + x1) / 2} y={344} textAnchor="middle" fill="#a6b0d6" fontSize="12">
                                Genotype dosage class (0/1/2)
                              </text>
                              <text x={26} y={(y0 + y1) / 2} textAnchor="middle" fill="#a6b0d6" fontSize="12" transform={`rotate(-90 26 ${(y0 + y1) / 2})`}>
                                {markerEffect.phenotype_field}
                              </text>
                            </>
                          );
                        })()}
                      </svg>
                      <div className="hint" style={{ marginTop: 6 }}>
                        Box: Q1-Q3, line: median, orange point: mean, whiskers: 1.5×IQR rule.
                      </div>
                    </>
                  ) : markerEffectError ? (
                    <div className="hint" style={{ color: "var(--danger)" }}>
                      Could not build boxplot: {markerEffectError}
                    </div>
                  ) : (
                    <div className="hint">Run mapping to compute a top-marker effect boxplot.</div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="hint" style={{ marginTop: 12 }}>
              Run mapping to see top associated markers.
            </div>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panelTitle">Recent Mapping History</div>
        {history.length ? (
          <PaginatedTable
            rows={history}
            columns={historyColumns}
            getRowKey={(r, idx) => `${r.created_at}-${r.marker_name}-${idx}`}
            initialSort={{ columnId: "created_at", dir: "desc" }}
            initialPageSize={25}
            searchableText={(r) => `${r.marker_name} ${r.phenotype_field} ${r.created_at}`}
            emptyLabel="No mapping runs yet."
          />
        ) : (
          <div className="hint">No mapping runs yet.</div>
        )}
      </section>
    </main>
  );
}
