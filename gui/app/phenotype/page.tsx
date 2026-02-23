"use client";

import React, { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { ProgressBar } from "../_components/ProgressBar";
import { PaginatedTable, type PaginatedTableColumn } from "../_components/PaginatedTable";
import { parseMiniFromFilename } from "../_lib/minirhizotron";

type AnalysisResult = {
  root_count: number;
  average_root_diameter: number;
  total_root_length: number;
  total_root_area: number;
  total_root_volume: number;
};

type BatchAnalyzeItem =
  | { filename: string; success: true; result: AnalysisResult & { mask_image_base64: string } }
  | { filename: string; success: false; error: string };

type BatchAnalyzeResponse = {
  results: BatchAnalyzeItem[];
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";

type PhenotypeMode = "standard" | "minirhizotron";

type ResultRow = {
  filename: string;
  ok: boolean;
  metrics?: AnalysisResult;
  error?: string;
  originalDataUrl?: string;
  maskDataUrl?: string;
};

type MiniMeta = {
  field?: string;
  tube_id?: string;
  genotype?: string;
  depth?: number;
  depth_length_cm?: number;
  timepoint?: string;
  session_label?: string;
  session_time?: string; // ISO
  camera_model?: string;
  camera_dpi?: number;
  pixel_to_cm?: number;
  image_size_cm?: string;
};

function toCsv(rows: Array<Record<string, any>>): string {
  const headers = Array.from(new Set(rows.flatMap((r) => Object.keys(r))));
  const escape = (v: any) => `"${String(v ?? "").replaceAll('"', '""')}"`;
  const lines = [headers.map(escape).join(",")];
  for (const r of rows) {
    lines.push(headers.map((h) => escape(r[h])).join(","));
  }
  return lines.join("\n");
}

function normalizeTubeId(raw: string): string {
  const v = (raw || "").trim();
  if (!v) return "";
  const m = v.match(/^t?(\d+)$/i);
  if (m?.[1]) return `T${m[1]}`;
  return v.toUpperCase();
}

function parseCsvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i]!;
    if (ch === '"') {
      if (inQuotes && text[i + 1] === '"') {
        cell += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === "," && !inQuotes) {
      row.push(cell);
      cell = "";
      continue;
    }
    if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && text[i + 1] === "\n") i += 1;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
      continue;
    }
    cell += ch;
  }
  row.push(cell);
  if (row.length > 1 || row[0] !== "") rows.push(row);
  return rows;
}

function parseTubeGenotypeMapCsv(text: string): Record<string, string> {
  const rows = parseCsvRows(text);
  if (!rows.length) throw new Error("CSV is empty");
  const head = rows[0]!.map((x) => (x || "").trim().toLowerCase());
  const tubeIdx = head.findIndex((h) => h === "tube id" || h === "tube_id" || h === "tube" || h === "tubeid");
  const genoIdx = head.findIndex((h) => h === "genotype" || h === "line" || h === "accession");
  if (tubeIdx < 0 || genoIdx < 0) {
    throw new Error("CSV must include Tube ID and Genotype columns");
  }
  const out: Record<string, string> = {};
  for (let i = 1; i < rows.length; i += 1) {
    const r = rows[i] ?? [];
    const tube = normalizeTubeId((r[tubeIdx] || "").trim());
    const geno = (r[genoIdx] || "").trim();
    if (!tube || !geno) continue;
    out[tube] = geno;
  }
  return out;
}

export default function Page() {
  const router = useRouter();
  const [runId, setRunId] = useState<string>("");
  const [thresholdArea, setThresholdArea] = useState<number>(50);
  const [scalingFactor, setScalingFactor] = useState<number>(1.0);
  const [mode, setMode] = useState<PhenotypeMode>("standard");
  const [sessionLabel, setSessionLabel] = useState<string>("session_1");
  const [sessionTimeLocal, setSessionTimeLocal] = useState<string>("");
  const [defaultTubeId, setDefaultTubeId] = useState<string>("");
  const [defaultGenotype, setDefaultGenotype] = useState<string>("");
  const [defaultTimepoint, setDefaultTimepoint] = useState<string>("T1");
  const [defaultDepthLengthCm, setDefaultDepthLengthCm] = useState<number>(10);
  const [cameraModel, setCameraModel] = useState<"ci600">("ci600");
  const [ci600Dpi, setCi600Dpi] = useState<number>(600);
  const [useManualPixelToCm, setUseManualPixelToCm] = useState<boolean>(false);
  const [manualPixelToCm, setManualPixelToCm] = useState<number>(2.54 / 600);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ResultRow[]>([]);
  const [rowMeta, setRowMeta] = useState<Record<string, MiniMeta>>({});
  const [previewFilename, setPreviewFilename] = useState<string | null>(null);
  const [tubeGenotypeMap, setTubeGenotypeMap] = useState<Record<string, string>>({});
  const [tubeMapFileName, setTubeMapFileName] = useState<string>("");

  React.useEffect(() => {
    const saved = localStorage.getItem("subterra_run_id");
    if (saved) setRunId(saved);
  }, []);

  React.useEffect(() => {
    localStorage.setItem("subterra_run_id", runId);
  }, [runId]);

  const okRows = useMemo(
    () =>
      results
        .filter((r) => r.ok && r.metrics)
        .map((r) => {
          if (mode !== "minirhizotron") {
            return { filename: r.filename, ...(r.metrics as AnalysisResult) };
          }
          const meta = rowMeta[r.filename] ?? {};
          return {
            filename: r.filename,
            field: meta.field ?? "",
            tube_id: meta.tube_id ?? "",
            genotype: meta.genotype ?? "",
            depth: meta.depth ?? "",
            depth_length_cm: meta.depth_length_cm ?? "",
            timepoint: meta.timepoint ?? "",
            session_label: meta.session_label ?? "",
            session_time: meta.session_time ?? "",
            ...(r.metrics as AnalysisResult)
          };
        }),
    [results, rowMeta, mode]
  );

  const preview = useMemo(
    () => results.find((r) => r.filename === previewFilename) ?? null,
    [results, previewFilename]
  );

  const resultColumns: Array<PaginatedTableColumn<ResultRow>> = (() => {
    const cols: Array<PaginatedTableColumn<ResultRow>> = [
      {
        id: "filename",
        header: "filename",
        sortValue: (r) => r.filename,
        cell: (r) => (
          <>
            <button className="btn" style={{ padding: "6px 8px" }} onClick={() => setPreviewFilename(r.filename)}>
              View
            </button>{" "}
            <span style={{ marginLeft: 8 }}>{r.filename}</span>
          </>
        )
      }
    ];

    if (mode === "minirhizotron") {
      cols.push(
        { id: "field", header: "field", sortValue: (r) => rowMeta[r.filename]?.field ?? "", cell: (r) => rowMeta[r.filename]?.field ?? "" },
        { id: "tube_id", header: "tube_id", sortValue: (r) => rowMeta[r.filename]?.tube_id ?? "", cell: (r) => rowMeta[r.filename]?.tube_id ?? "" },
        { id: "genotype", header: "genotype", sortValue: (r) => rowMeta[r.filename]?.genotype ?? "", cell: (r) => rowMeta[r.filename]?.genotype ?? "" },
        { id: "depth", header: "depth", sortValue: (r) => rowMeta[r.filename]?.depth ?? null, cell: (r) => String(rowMeta[r.filename]?.depth ?? "") },
        { id: "timepoint", header: "timepoint", sortValue: (r) => rowMeta[r.filename]?.timepoint ?? "", cell: (r) => rowMeta[r.filename]?.timepoint ?? "" }
      );
    }

    cols.push(
      { id: "root_count", header: "root_count", sortValue: (r) => r.metrics?.root_count ?? null, cell: (r) => r.metrics?.root_count ?? "—" },
      {
        id: "average_root_diameter",
        header: "average_root_diameter",
        sortValue: (r) => r.metrics?.average_root_diameter ?? null,
        cell: (r) => r.metrics?.average_root_diameter ?? "—"
      },
      {
        id: "total_root_length",
        header: "total_root_length",
        sortValue: (r) => r.metrics?.total_root_length ?? null,
        cell: (r) => r.metrics?.total_root_length ?? "—"
      },
      { id: "total_root_area", header: "total_root_area", sortValue: (r) => r.metrics?.total_root_area ?? null, cell: (r) => r.metrics?.total_root_area ?? "—" },
      {
        id: "total_root_volume",
        header: "total_root_volume",
        sortValue: (r) => r.metrics?.total_root_volume ?? null,
        cell: (r) => r.metrics?.total_root_volume ?? "—"
      },
      {
        id: "status",
        header: "status",
        sortValue: (r) => (r.ok ? "ok" : "error"),
        cell: (r) => (r.ok ? "ok" : <span style={{ color: "var(--danger)" }}>{r.error}</span>)
      },
      {
        id: "actions",
        header: "actions",
        cell: (r) => (
          <button
            className="btn btnPrimary"
            style={{ padding: "6px 8px" }}
            onClick={() => sendToAnnotation(r.filename)}
            disabled={!r.ok || !r.originalDataUrl || !r.maskDataUrl}
          >
            Annotate
          </button>
        )
      }
    );

    return cols;
  })();

  const resultsSearchableText = (r: ResultRow) => {
    if (mode !== "minirhizotron") return `${r.filename} ${r.ok ? "ok" : "error"} ${r.error ?? ""}`;
    const m = rowMeta[r.filename] ?? {};
    return `${r.filename} ${m.field ?? ""} ${m.tube_id ?? ""} ${m.genotype ?? ""} ${m.depth ?? ""} ${m.timepoint ?? ""} ${r.ok ? "ok" : "error"} ${r.error ?? ""}`;
  };

  const derivedPixelToCm = useMemo(() => {
    if (useManualPixelToCm) return manualPixelToCm;
    // CI-600 supports 100–600 DPI scanning; use selected DPI to convert pixels→cm
    const dpi = Math.max(1, ci600Dpi);
    return 2.54 / dpi;
  }, [useManualPixelToCm, manualPixelToCm, ci600Dpi]);

  const effectiveScalingFactor = mode === "minirhizotron" ? derivedPixelToCm : scalingFactor;

  function genotypeForTube(tubeId?: string): string | undefined {
    const key = normalizeTubeId(tubeId || "");
    if (!key) return undefined;
    return tubeGenotypeMap[key];
  }

  async function loadTubeMapFile(file: File) {
    try {
      const text = await file.text();
      const parsed = parseTubeGenotypeMapCsv(text);
      setTubeGenotypeMap(parsed);
      setTubeMapFileName(file.name);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function applyTubeMapToRows() {
    if (!Object.keys(tubeGenotypeMap).length) return;
    setRowMeta((prev) => {
      const next: Record<string, MiniMeta> = { ...prev };
      for (const [fn, meta] of Object.entries(prev)) {
        const g = genotypeForTube(meta.tube_id);
        if (!g) continue;
        next[fn] = { ...meta, genotype: g };
      }
      return next;
    });
  }

  async function run(files: File[]) {
    setError(null);
    setRunning(true);
    try {
      const form = new FormData();
      for (const f of files) form.append("files", f, f.name);
      if (runId.trim()) form.append("run_id", runId.trim());

      const sessionIso = sessionTimeLocal ? new Date(sessionTimeLocal).toISOString() : new Date().toISOString();
      if (mode === "minirhizotron") {
        for (const f of files) {
          const parsed = parseMiniFromFilename(f.name);
          const tubeId = parsed.tube_id ?? (defaultTubeId || undefined);
          const mappedGenotype = genotypeForTube(tubeId);
          const meta: MiniMeta = {
            field: parsed.field ?? undefined,
            tube_id: tubeId,
            genotype: mappedGenotype ?? (defaultGenotype || undefined),
            depth: parsed.depth ?? undefined,
            depth_length_cm: defaultDepthLengthCm,
            timepoint: defaultTimepoint || undefined,
            session_label: sessionLabel || undefined,
            session_time: sessionIso
          };
          const camera = {
            camera_model: cameraModel === "ci600" ? "CID CI-600 In-Situ Root Imager" : "Unknown",
            camera_dpi: ci600Dpi,
            pixel_to_cm: derivedPixelToCm,
            image_size_cm: "21.6 x 19.6"
          };
          form.append("metadata_json", JSON.stringify({ phenotyping_mode: mode, minirhizotron: meta, camera }));
        }
      }

      const params = new URLSearchParams({
        model_type: "unet",
        threshold_area: String(thresholdArea),
        scaling_factor: String(effectiveScalingFactor)
      });

      const res = await fetch(`${API_BASE_URL}/batch-analyze?${params}`, { method: "POST", body: form });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as BatchAnalyzeResponse;
      const mapped = data.results.map((it) => {
        if (!it.success) return { filename: it.filename, ok: false, error: it.error };
        const { mask_image_base64, ...metrics } = it.result as any;
        const originalDataUrl = (it.result as any).original_image_base64 as string | undefined;
        return { filename: it.filename, ok: true, metrics, originalDataUrl, maskDataUrl: mask_image_base64 };
      });
      setResults(mapped);
      if (mode === "minirhizotron") {
        const nextMeta: Record<string, MiniMeta> = {};
        for (const r of mapped) {
          if (!r.ok) continue;
          const parsed = parseMiniFromFilename(r.filename);
          const tubeId = parsed.tube_id ?? (defaultTubeId || undefined);
          const mappedGenotype = genotypeForTube(tubeId);
          nextMeta[r.filename] = {
            field: parsed.field ?? undefined,
            tube_id: tubeId,
            genotype: mappedGenotype ?? (defaultGenotype || undefined),
            depth: parsed.depth ?? undefined,
            depth_length_cm: defaultDepthLengthCm,
            timepoint: defaultTimepoint || undefined,
            session_label: sessionLabel || undefined,
            session_time: sessionIso,
            camera_model: cameraModel === "ci600" ? "CID CI-600 In-Situ Root Imager" : undefined,
            camera_dpi: ci600Dpi,
            pixel_to_cm: derivedPixelToCm,
            image_size_cm: "21.6 x 19.6"
          };
        }
        setRowMeta(nextMeta);
      } else {
        setRowMeta({});
      }
      setPreviewFilename(mapped.find((r) => r.ok)?.filename ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  function sendToAnnotation(filename: string) {
    const r = results.find((x) => x.filename === filename);
    if (!r || !r.ok || !r.originalDataUrl || !r.maskDataUrl) return;

    const id = crypto.randomUUID();
    window.__subterraHandoff = window.__subterraHandoff ?? {};
    const meta = rowMeta[filename] ?? {};
    window.__subterraHandoff[id] = {
      source: "phenotype",
      createdAt: new Date().toISOString(),
      items: [
        {
          filename: r.filename,
          originalDataUrl: r.originalDataUrl,
          maskDataUrl: r.maskDataUrl,
          metrics: r.metrics ?? null,
          meta
        }
      ]
    };
    router.push(`/annotate?handoff=${encodeURIComponent(id)}`);
  }


  function downloadCsv() {
    const csv = toCsv(okRows);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "phenotyping_metrics.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main style={{ display: "grid", gap: 16 }}>
      <header className="panel">
        <h1 style={{ margin: 0, fontSize: 18 }}>Phenotyping</h1>
        <p style={{ margin: "6px 0 0", color: "var(--muted)", fontSize: 13 }}>
          Run batch inference (U-Net) and export root metrics as CSV.
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
            <div className="hint">Phenotype rows are tagged with this `run_id` (used in paper exports).</div>
          </div>
        </div>
      </header>

      <section className="panel">
        <div className="panelTitle">Batch Analyze</div>
        <ProgressBar active={running} label={running ? "Running…" : undefined} />
        <div className="row">
          <div className="field">
            <div className="label">Approach</div>
            <select className="select" value={mode} onChange={(e) => setMode(e.target.value as PhenotypeMode)}>
              <option value="standard">Standard</option>
              <option value="minirhizotron">Mini-rhizotron</option>
            </select>
          </div>
          <div className="field">
            <div className="label">Model</div>
            <input className="input" value="U-Net (fixed)" readOnly />
          </div>
          <div className="field">
            <div className="label">Area Threshold</div>
            <input className="input" type="number" min={0} value={thresholdArea} onChange={(e) => setThresholdArea(Number(e.target.value))} />
          </div>
          <div className="field">
            <div className="label">Scaling Factor</div>
            <input
              className="input"
              type="number"
              step={0.0001}
              value={effectiveScalingFactor}
              onChange={(e) => setScalingFactor(Number(e.target.value))}
              disabled={mode === "minirhizotron"}
            />
          </div>
          <div className="field">
            <div className="label">Images</div>
            <input
              className="input"
              type="file"
              accept="image/*"
              multiple
              disabled={running}
              onChange={(e) => {
                const f = e.currentTarget.files;
                if (!f || f.length === 0) return;
                void run(Array.from(f));
              }}
            />
          </div>
          <button className="btn btnPrimary" onClick={downloadCsv} disabled={okRows.length === 0}>
            Download CSV
          </button>
        </div>
        {error ? <div style={{ marginTop: 10, color: "var(--danger)", fontSize: 13 }}>{error}</div> : null}
      </section>

      {mode === "minirhizotron" ? (
        <section className="panel">
          <div className="panelTitle">Mini-rhizotron</div>
          <div className="hint">
            Record tube-based time series. Filename parsing supports CI-600 patterns like `CS_T474_L007.PNG` (field, tube_id, depth) or tokens like `tube01` and `depth7`/`d7`. These fields are stored in the analysis record.
          </div>
          <div className="panel" style={{ marginTop: 12, padding: 12 }}>
            <div className="panelTitle">Camera + scaling</div>
            <div className="hint">
              Default camera: CID Bio-Science CI-600. Specs: image size 21.6 × 19.6 cm, scan resolution 100–600 DPI.
              Scaling is computed as `pixel_to_cm = 2.54 / DPI` unless manual override is enabled.
            </div>
            <div className="row" style={{ marginTop: 10 }}>
              <div className="field">
                <div className="label">Camera</div>
                <select className="select" value={cameraModel} onChange={(e) => setCameraModel(e.target.value as "ci600")}>
                  <option value="ci600">CID CI-600 In-Situ Root Imager</option>
                </select>
              </div>
              <div className="field">
                <div className="label">Scan DPI</div>
                <select className="select" value={ci600Dpi} onChange={(e) => setCi600Dpi(Number(e.target.value))}>
                  {[100, 200, 300, 400, 500, 600].map((d) => (
                    <option key={d} value={d}>
                      {d} DPI
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <div className="label">Manual pixel→cm</div>
                <select className="select" value={useManualPixelToCm ? "manual" : "auto"} onChange={(e) => setUseManualPixelToCm(e.target.value === "manual")}>
                  <option value="auto">Auto (from DPI)</option>
                  <option value="manual">Manual</option>
                </select>
              </div>
              <div className="field">
                <div className="label">pixel_to_cm</div>
                <input
                  className="input"
                  type="number"
                  step={0.000001}
                  value={useManualPixelToCm ? manualPixelToCm : derivedPixelToCm}
                  onChange={(e) => setManualPixelToCm(Number(e.target.value))}
                  disabled={!useManualPixelToCm}
                />
              </div>
            </div>
            <div className="hint" style={{ marginTop: 8 }}>
              Current scaling_factor sent to API: {derivedPixelToCm.toFixed(6)} cm/pixel (outputs are in cm, cm², cm³). Set this before uploading/running.
            </div>
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <div className="field">
              <div className="label">Session label</div>
              <input className="input" value={sessionLabel} onChange={(e) => setSessionLabel(e.target.value)} />
            </div>
            <div className="field">
              <div className="label">Session time</div>
              <input className="input" type="datetime-local" value={sessionTimeLocal} onChange={(e) => setSessionTimeLocal(e.target.value)} />
            </div>
            <div className="field">
              <div className="label">Timepoint</div>
              <input className="input" value={defaultTimepoint} onChange={(e) => setDefaultTimepoint(e.target.value)} />
            </div>
            <div className="field">
              <div className="label">Depth length (cm)</div>
              <input
                className="input"
                type="number"
                min={0}
                step={0.5}
                value={defaultDepthLengthCm}
                onChange={(e) => setDefaultDepthLengthCm(Number(e.target.value))}
              />
            </div>
            <div className="field">
              <div className="label">Default tube id</div>
              <input className="input" value={defaultTubeId} onChange={(e) => setDefaultTubeId(e.target.value)} placeholder="01" />
            </div>
            <div className="field">
              <div className="label">Default genotype</div>
              <input className="input" value={defaultGenotype} onChange={(e) => setDefaultGenotype(e.target.value)} placeholder="A" />
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, alignItems: "center", gap: 10 }}>
            <div className="field" style={{ minWidth: 280 }}>
              <div className="label">Tube→Genotype CSV (optional)</div>
              <input
                className="input"
                type="file"
                accept=".csv,text/csv"
                onChange={(e) => {
                  const f = e.currentTarget.files?.[0];
                  if (!f) return;
                  void loadTubeMapFile(f);
                }}
              />
            </div>
            <button className="btn" onClick={applyTubeMapToRows} disabled={!Object.keys(tubeGenotypeMap).length || !Object.keys(rowMeta).length}>
              Apply map to loaded rows
            </button>
            <div className="hint">
              {tubeMapFileName
                ? `${tubeMapFileName}: ${Object.keys(tubeGenotypeMap).length} tube labels loaded`
                : "Expected columns: Tube ID, Genotype"}
            </div>
          </div>

          {preview && preview.ok ? (
            <div className="panel" style={{ marginTop: 12, padding: 12 }}>
              <div className="panelTitle">Selected image metadata</div>
              <div className="row" style={{ marginTop: 10 }}>
                <div className="field">
                  <div className="label">Field / farm</div>
                  <input
                    className="input"
                    value={rowMeta[preview.filename]?.field ?? ""}
                    onChange={(e) =>
                      setRowMeta((prev) => ({
                        ...prev,
                        [preview.filename]: { ...(prev[preview.filename] ?? {}), field: e.target.value }
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <div className="label">Tube id</div>
                  <input
                    className="input"
                    value={rowMeta[preview.filename]?.tube_id ?? ""}
                    onChange={(e) =>
                      setRowMeta((prev) => {
                        const nextTube = e.target.value;
                        const mappedGenotype = genotypeForTube(nextTube);
                        return {
                          ...prev,
                          [preview.filename]: {
                            ...(prev[preview.filename] ?? {}),
                            tube_id: nextTube,
                            genotype: mappedGenotype ?? (prev[preview.filename]?.genotype ?? "")
                          }
                        };
                      })
                    }
                  />
                </div>
                <div className="field">
                  <div className="label">Genotype</div>
                  <input
                    className="input"
                    value={rowMeta[preview.filename]?.genotype ?? ""}
                    onChange={(e) =>
                      setRowMeta((prev) => ({
                        ...prev,
                        [preview.filename]: { ...(prev[preview.filename] ?? {}), genotype: e.target.value }
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <div className="label">Depth</div>
                  <input
                    className="input"
                    type="number"
                    min={0}
                    step={1}
                    value={rowMeta[preview.filename]?.depth ?? ""}
                    onChange={(e) =>
                      setRowMeta((prev) => ({
                        ...prev,
                        [preview.filename]: { ...(prev[preview.filename] ?? {}), depth: Number(e.target.value) }
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <div className="label">Depth length (cm)</div>
                  <input
                    className="input"
                    type="number"
                    min={0}
                    step={0.5}
                    value={rowMeta[preview.filename]?.depth_length_cm ?? ""}
                    onChange={(e) =>
                      setRowMeta((prev) => ({
                        ...prev,
                        [preview.filename]: { ...(prev[preview.filename] ?? {}), depth_length_cm: Number(e.target.value) }
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <div className="label">Timepoint</div>
                  <input
                    className="input"
                    value={rowMeta[preview.filename]?.timepoint ?? ""}
                    onChange={(e) =>
                      setRowMeta((prev) => ({
                        ...prev,
                        [preview.filename]: { ...(prev[preview.filename] ?? {}), timepoint: e.target.value }
                      }))
                    }
                  />
                </div>
              </div>
              <div className="hint" style={{ marginTop: 8 }}>
                Note: editing here affects CSV export and annotation handoff; the analysis record metadata is set at run time.
              </div>
            </div>
          ) : null}
        </section>
      ) : null}

      <section className="panel">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <div className="panelTitle">Results</div>
          <button className="btn btnPrimary" onClick={downloadCsv} disabled={okRows.length === 0}>
            Download CSV
          </button>
        </div>
        {results.length === 0 ? (
          <div className="hint">Upload images to see metrics.</div>
        ) : (
          <div style={{ display: "grid", gap: 12 }}>
            <PaginatedTable
              rows={results}
              columns={resultColumns}
              getRowKey={(r) => r.filename}
              initialPageSize={25}
              searchableText={resultsSearchableText}
              isRowActive={(r) => r.filename === previewFilename}
              emptyLabel="No results."
              footerLeft={<div className="hint">Tip: click “View” to update the preview below.</div>}
            />

            {preview && preview.ok && preview.originalDataUrl && preview.maskDataUrl ? (
              <div className="panel" style={{ padding: 12 }}>
                <div className="panelTitle">Preview: original vs predicted mask</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <div className="canvasFrame" style={{ padding: 10 }}>
                    <div className="hint" style={{ marginBottom: 6 }}>
                      Original
                    </div>
                    <img src={preview.originalDataUrl} alt={`${preview.filename} original`} style={{ width: "100%", height: "auto", borderRadius: 12, border: "1px solid var(--border)" }} />
                  </div>
                  <div className="canvasFrame" style={{ padding: 10 }}>
                    <div className="hint" style={{ marginBottom: 6 }}>
                      Predicted Mask
                    </div>
                    <img src={preview.maskDataUrl} alt={`${preview.filename} mask`} style={{ width: "100%", height: "auto", borderRadius: 12, border: "1px solid var(--border)" }} />
                  </div>
                </div>
                <div className="row" style={{ marginTop: 12 }}>
                  <button className="btn btnPrimary" onClick={() => sendToAnnotation(preview.filename)}>
                    Open in Annotation (with suggested mask)
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        )}
      </section>
    </main>
  );
}
