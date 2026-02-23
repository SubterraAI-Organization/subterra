"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { ProgressBar } from "../_components/ProgressBar";
import { parseMiniFromFilename } from "../_lib/minirhizotron";

type AnalysisResult = {
  root_count: number;
  average_root_diameter: number;
  total_root_length: number;
  total_root_area: number;
  total_root_volume: number;
  mask_image_base64: string;
  original_image_base64?: string | null;
};

type BatchAnalyzeItem =
  | { filename: string; success: true; result: AnalysisResult }
  | { filename: string; success: false; error: string };

type BatchAnalyzeResponse = {
  results: BatchAnalyzeItem[];
};

type ItemStatus = "ready" | "saving" | "saved" | "error";

type CaptureMode = "standard" | "minirhizotron";

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

type ImageItem = {
  id: string;
  file: File;
  filename: string;
  originalDataUrl: string;
  maskDataUrl: string;
  metrics?: Omit<AnalysisResult, "mask_image_base64" | "original_image_base64">;
  meta?: Record<string, any> | null;
  status: ItemStatus;
  error?: string;
  savedAnnotationId?: string;
  dirty: boolean;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";

function formatNumber(value: number) {
  if (Number.isNaN(value)) return "—";
  return value.toFixed(3);
}

async function toPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob((b) => resolve(b), "image/png")
  );
  if (!blob) throw new Error("Failed to export PNG");
  return blob;
}

export default function AnnotateClient() {
  const searchParams = useSearchParams();
  const [thresholdArea, setThresholdArea] = useState<number>(50);
  const [scalingFactor, setScalingFactor] = useState<number>(1.0);
  const [runId, setRunId] = useState<string>("");

  const [captureMode, setCaptureMode] = useState<CaptureMode>("standard");
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

  const [items, setItems] = useState<ImageItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [importNotice, setImportNotice] = useState<string | null>(null);
  const [savingAll, setSavingAll] = useState<{ active: boolean; done: number; total: number }>({
    active: false,
    done: 0,
    total: 0
  });

  const selectedItem = useMemo(
    () => items.find((i) => i.id === selectedId) ?? null,
    [items, selectedId]
  );

  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const undoStackRef = useRef<ImageData[]>([]);
  const strokeChangedRef = useRef(false);

  const [brushSize, setBrushSize] = useState<number>(18);
  const [tool, setTool] = useState<"add" | "erase">("add");
  const [isDrawing, setIsDrawing] = useState(false);

  const itemsRef = useRef<ImageItem[]>([]);
  useEffect(() => {
    itemsRef.current = items;
  }, [items]);

  useEffect(() => {
    const saved = localStorage.getItem("subterra_run_id");
    if (saved) setRunId(saved);
  }, []);

  useEffect(() => {
    localStorage.setItem("subterra_run_id", runId);
  }, [runId]);

  const derivedPixelToCm = useMemo(() => {
    if (useManualPixelToCm) return manualPixelToCm;
    const dpi = Math.max(1, ci600Dpi);
    return 2.54 / dpi;
  }, [useManualPixelToCm, manualPixelToCm, ci600Dpi]);

  const effectiveScalingFactor = captureMode === "minirhizotron" ? derivedPixelToCm : scalingFactor;

  function buildDefaultMeta(filename: string): MiniMeta | null {
    if (captureMode !== "minirhizotron") return null;

    const parsed = parseMiniFromFilename(filename);
    const sessionIso = sessionTimeLocal ? new Date(sessionTimeLocal).toISOString() : new Date().toISOString();
    return {
      field: parsed.field ?? undefined,
      tube_id: parsed.tube_id ?? (defaultTubeId || undefined),
      genotype: defaultGenotype || undefined,
      depth: parsed.depth ?? undefined,
      depth_length_cm: defaultDepthLengthCm,
      timepoint: defaultTimepoint || undefined,
      session_label: sessionLabel || undefined,
      session_time: sessionIso,
      camera_model: cameraModel === "ci600" ? "CID CI-600 In-Situ Root Imager" : "Unknown",
      camera_dpi: ci600Dpi,
      pixel_to_cm: derivedPixelToCm,
      image_size_cm: "21.6 x 19.6"
    };
  }

  function patchSelectedMeta(patch: Partial<MiniMeta>) {
    if (!selectedId) return;
    setItems((prev) =>
      prev.map((it) => {
        if (it.id !== selectedId) return it;
        const next: MiniMeta = { ...(it.meta ?? {}), ...patch };
        return { ...it, meta: next };
      })
    );
  }

  async function dataUrlToFile(dataUrl: string, filename: string): Promise<File> {
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    const type = blob.type || "image/png";
    return new File([blob], filename, { type });
  }

  useEffect(() => {
    const handoffId = searchParams.get("handoff");
    if (!handoffId) return;
    if (typeof window === "undefined") return;

    const payload = window.__subterraHandoff?.[handoffId];
    if (!payload) {
      setImportNotice("No handoff payload found (try sending again from Phenotyping).");
      return;
    }

    setImporting(true);
    setImportNotice(`Imported from ${payload.source}`);

    (async () => {
      const next: ImageItem[] = [];
      for (const it of payload.items) {
        const file = await dataUrlToFile(it.originalDataUrl, it.filename);
        next.push({
          id: crypto.randomUUID(),
          file,
          filename: it.filename,
          originalDataUrl: it.originalDataUrl,
          maskDataUrl: it.maskDataUrl,
          metrics: (it.metrics ?? undefined) as any,
          meta: (it.meta ?? null) as any,
          status: "ready",
          dirty: false
        });
      }
      setItems(next);
      setSelectedId(next[0]?.id ?? null);
    })()
      .catch((e) => setAnalyzeError(e instanceof Error ? e.message : String(e)))
      .finally(() => setImporting(false));
  }, [searchParams]);

  async function analyzeFiles(files: File[]) {
    setAnalyzeError(null);
    setAnalyzing(true);
    try {
      const form = new FormData();
      for (const file of files) form.append("files", file, file.name);
      if (runId.trim()) form.append("run_id", runId.trim());

      const params = new URLSearchParams({
        model_type: "unet",
        threshold_area: String(thresholdArea),
        scaling_factor: String(effectiveScalingFactor)
      });

      const response = await fetch(`${API_BASE_URL}/batch-analyze?${params}`, {
        method: "POST",
        body: form
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `HTTP ${response.status}`);
      }

      const data = (await response.json()) as BatchAnalyzeResponse;
      const nextItems: ImageItem[] = data.results.map((result, index) => {
        const file = files[index];
        if (!file) {
          throw new Error("Internal error: mismatched batch response");
        }

        const meta = buildDefaultMeta(result.filename || file.name);
        if (!result.success) {
          return {
            id: crypto.randomUUID(),
            file,
            filename: result.filename,
            originalDataUrl: URL.createObjectURL(file),
            maskDataUrl: "",
            meta,
            status: "error",
            error: result.error,
            dirty: false
          };
        }

        const { mask_image_base64, original_image_base64, ...metrics } = result.result;
        return {
          id: crypto.randomUUID(),
          file,
          filename: result.filename,
          originalDataUrl: original_image_base64 ?? URL.createObjectURL(file),
          maskDataUrl: mask_image_base64,
          metrics,
          meta,
          status: "ready",
          dirty: false
        };
      });

      setItems(nextItems);
      setSelectedId(nextItems.find((i) => i.status !== "error")?.id ?? null);
    } catch (err) {
      setAnalyzeError(err instanceof Error ? err.message : String(err));
    } finally {
      setAnalyzing(false);
    }
  }

  function clearSession() {
    setItems((prev) => {
      for (const item of prev) {
        if (item.originalDataUrl.startsWith("blob:")) URL.revokeObjectURL(item.originalDataUrl);
      }
      return [];
    });
    setSelectedId(null);
    undoStackRef.current = [];
  }

  function renderOverlay() {
    const maskCanvas = maskCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    if (!maskCanvas || !overlayCanvas) return;

    const ctx = overlayCanvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    ctx.drawImage(maskCanvas, 0, 0);
    ctx.globalCompositeOperation = "source-in";
    ctx.fillStyle = "rgba(255, 60, 60, 0.45)";
    ctx.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    ctx.globalCompositeOperation = "source-over";
  }

  useEffect(() => {
    const item = selectedItem;
    undoStackRef.current = [];
    if (!item || !item.maskDataUrl) return;

    const originalDataUrl = item.originalDataUrl;
    const maskDataUrl = item.maskDataUrl;

    let cancelled = false;

    async function init() {
      const maskCanvas = maskCanvasRef.current;
      const overlayCanvas = overlayCanvasRef.current;
      if (!maskCanvas || !overlayCanvas) return;

      const original = new Image();
      original.src = originalDataUrl;
      await original.decode();

      const width = original.naturalWidth;
      const height = original.naturalHeight;
      if (!width || !height) throw new Error("Failed to read image dimensions");

      maskCanvas.width = width;
      maskCanvas.height = height;
      overlayCanvas.width = width;
      overlayCanvas.height = height;

      const maskImg = new Image();
      maskImg.src = maskDataUrl;
      await maskImg.decode();

      const temp = document.createElement("canvas");
      temp.width = width;
      temp.height = height;
      const tempCtx = temp.getContext("2d");
      const maskCtx = maskCanvas.getContext("2d");
      if (!tempCtx || !maskCtx) return;

      tempCtx.drawImage(maskImg, 0, 0, width, height);
      const src = tempCtx.getImageData(0, 0, width, height);
      const dst = maskCtx.createImageData(width, height);

      for (let i = 0; i < src.data.length; i += 4) {
        const v = src.data[i] ?? 0;
        const a = src.data[i + 3] ?? 0;
        const on = a > 127 && v > 127;
        dst.data[i] = 255;
        dst.data[i + 1] = 255;
        dst.data[i + 2] = 255;
        dst.data[i + 3] = on ? 255 : 0;
      }

      if (cancelled) return;
      maskCtx.putImageData(dst, 0, 0);
      renderOverlay();
    }

    init().catch((e) => {
      if (!cancelled) console.error(e);
    });

    return () => {
      cancelled = true;
    };
  }, [selectedItem]);

  function canvasPointFromEvent(e: React.PointerEvent<HTMLCanvasElement>) {
    const overlayCanvas = overlayCanvasRef.current;
    if (!overlayCanvas) return null;
    const rect = overlayCanvas.getBoundingClientRect();
    const scaleX = overlayCanvas.width / rect.width;
    const scaleY = overlayCanvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    return { x, y };
  }

  function stampAt(x: number, y: number) {
    const maskCanvas = maskCanvasRef.current;
    if (!maskCanvas) return;
    const ctx = maskCanvas.getContext("2d");
    if (!ctx) return;

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (tool === "erase") {
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "rgba(0,0,0,1)";
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = "rgba(255,255,255,1)";
    }

    ctx.beginPath();
    ctx.arc(x, y, Math.max(1, brushSize / 2), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    renderOverlay();
    strokeChangedRef.current = true;
  }

  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    const p = canvasPointFromEvent(e);
    if (!p) return;
    const maskCanvas = maskCanvasRef.current;
    const ctx = maskCanvas?.getContext("2d");
    if (!maskCanvas || !ctx) return;

    const snapshot = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    undoStackRef.current.push(snapshot);
    undoStackRef.current = undoStackRef.current.slice(-10);

    strokeChangedRef.current = false;
    setIsDrawing(true);
    e.currentTarget.setPointerCapture(e.pointerId);
    stampAt(p.x, p.y);
  }

  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!isDrawing) return;
    const p = canvasPointFromEvent(e);
    if (!p) return;
    stampAt(p.x, p.y);
  }

  function onPointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
    setIsDrawing(false);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }
    const maskCanvas = maskCanvasRef.current;
    if (maskCanvas && selectedId && strokeChangedRef.current) {
      const nextMaskUrl = maskCanvas.toDataURL("image/png");
      setItems((prev) =>
        prev.map((it) =>
          it.id === selectedId ? { ...it, dirty: true, maskDataUrl: nextMaskUrl } : it
        )
      );
    }
  }

  function undo() {
    const maskCanvas = maskCanvasRef.current;
    const ctx = maskCanvas?.getContext("2d");
    if (!maskCanvas || !ctx) return;
    const snapshot = undoStackRef.current.pop();
    if (!snapshot) return;
    ctx.putImageData(snapshot, 0, 0);
    renderOverlay();
    if (selectedId) {
      const nextMaskUrl = maskCanvas.toDataURL("image/png");
      setItems((prev) =>
        prev.map((it) =>
          it.id === selectedId ? { ...it, dirty: true, maskDataUrl: nextMaskUrl } : it
        )
      );
    }
  }

  async function exportBinaryMaskPng(item: ImageItem): Promise<Blob> {
    const original = new Image();
    original.src = item.originalDataUrl;
    await original.decode();

    const width = original.naturalWidth;
    const height = original.naturalHeight;
    if (!width || !height) throw new Error("Failed to read image dimensions");

    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = width;
    exportCanvas.height = height;
    const exportCtx = exportCanvas.getContext("2d");
    if (!exportCtx) throw new Error("Canvas unsupported");

    exportCtx.fillStyle = "black";
    exportCtx.fillRect(0, 0, width, height);

    if (item.id === selectedId && maskCanvasRef.current) {
      exportCtx.drawImage(maskCanvasRef.current, 0, 0, width, height);
    } else if (item.maskDataUrl) {
      const maskImg = new Image();
      maskImg.src = item.maskDataUrl;
      await maskImg.decode();
      exportCtx.drawImage(maskImg, 0, 0, width, height);
    }

    return await toPngBlob(exportCanvas);
  }

  async function saveItem(itemId: string) {
    const item = itemsRef.current.find((it) => it.id === itemId);
    if (!item) return;

    setItems((prev) => prev.map((it) => (it.id === item.id ? { ...it, status: "saving" } : it)));
    try {
      const inferredScalingFactor =
        item.meta && typeof (item.meta as any).pixel_to_cm === "number"
          ? Number((item.meta as any).pixel_to_cm)
          : scalingFactor;
      const maskBlob = await exportBinaryMaskPng(item);
      const maskName = `${item.filename.replace(/\.[^.]+$/, "")}_mask.png`;

      const form = new FormData();
      form.append("image", item.file, item.file.name);
      form.append("mask", maskBlob, maskName);
      form.append("original_filename", item.filename);
      if (runId.trim()) form.append("run_id", runId.trim());
      form.append(
        "metadata_json",
        JSON.stringify({
          source: "subterra-hil-gui",
          model_type: "unet",
          threshold_area: thresholdArea,
          scaling_factor: inferredScalingFactor,
          metrics: item.metrics ?? null,
          corrected: item.dirty,
          capture_mode: captureMode,
          meta: item.meta ?? null
        })
      );

      const response = await fetch(`${API_BASE_URL}/annotations`, { method: "POST", body: form });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `HTTP ${response.status}`);
      }
      const saved = (await response.json()) as { annotation_id: string };

      setItems((prev) =>
        prev.map((it) =>
          it.id === item.id
            ? { ...it, status: "saved", savedAnnotationId: saved.annotation_id }
            : it
        )
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setItems((prev) =>
        prev.map((it) => (it.id === item.id ? { ...it, status: "error", error: message } : it))
      );
    }
  }

  async function saveAll() {
    const candidates = itemsRef.current.filter((it) => it.status !== "error");
    const total = candidates.length;
    setSavingAll({ active: true, done: 0, total });
    let done = 0;
    for (const it of candidates) {
      if (it.status !== "saved") {
        await saveItem(it.id);
      }
      done += 1;
      setSavingAll({ active: true, done, total });
    }
    setSavingAll({ active: false, done: total, total });
  }

  return (
    <main style={{ display: "grid", gap: 16 }}>
      <header className="panel">
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 18 }}>Annotation</h1>
            <p style={{ margin: "6px 0 0", color: "var(--muted)", fontSize: 13 }}>
              Upload images → get U-Net masks → correct with brush → save (image, mask) pairs for retraining.
            </p>
          </div>
          <div className="hint">API: {API_BASE_URL}</div>
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
            <div className="hint">Saved annotations are tagged with this `run_id` for paper exports.</div>
          </div>
        </div>
        {importNotice ? <div className="hint" style={{ marginTop: 8 }}>{importNotice}</div> : null}
      </header>

      <section className="panel">
        <div className="panelTitle">1) Batch Analyze (U-Net)</div>
        <ProgressBar active={importing || analyzing} label={importing ? "Loading handoff…" : analyzing ? "Analyzing…" : undefined} />
        <div className="row">
          <div className="field">
            <div className="label">Model</div>
            <input className="input" value="U-Net (fixed)" readOnly />
          </div>

          <div className="field">
            <div className="label">Area Threshold</div>
            <input
              className="input"
              type="number"
              value={thresholdArea}
              min={0}
              step={1}
              onChange={(e) => setThresholdArea(Number(e.target.value))}
            />
          </div>

          <div className="field">
            <div className="label">Scaling Factor</div>
            <input
              className="input"
              type="number"
              value={effectiveScalingFactor}
              step={0.01}
              disabled={captureMode === "minirhizotron"}
              onChange={(e) => setScalingFactor(Number(e.target.value))}
            />
          </div>

          <div className="field">
            <div className="label">Images</div>
            <input
              className="input"
              type="file"
              accept="image/*"
              multiple
              disabled={analyzing}
              onChange={(e) => {
                const f = e.currentTarget.files;
                if (!f || f.length === 0) return;
                void analyzeFiles(Array.from(f));
              }}
            />
          </div>

          <button className="btn btnDanger" onClick={clearSession} disabled={analyzing || items.length === 0}>
            Clear
          </button>
        </div>

        <div className="panel" style={{ marginTop: 12, padding: 12 }}>
          <div className="panelTitle">Capture Metadata (optional)</div>
          <div className="row">
            <div className="field">
              <div className="label">Mode</div>
              <select className="select" value={captureMode} onChange={(e) => setCaptureMode(e.target.value as CaptureMode)}>
                <option value="standard">Standard</option>
                <option value="minirhizotron">Minirhizotron</option>
              </select>
            </div>

            {captureMode === "minirhizotron" ? (
              <>
                <div className="field">
                  <div className="label">Session Label</div>
                  <input className="input" value={sessionLabel} onChange={(e) => setSessionLabel(e.target.value)} />
                </div>
                <div className="field">
                  <div className="label">Session Time</div>
                  <input className="input" type="datetime-local" value={sessionTimeLocal} onChange={(e) => setSessionTimeLocal(e.target.value)} />
                </div>
                <div className="field">
                  <div className="label">Default Tube ID</div>
                  <input className="input" value={defaultTubeId} onChange={(e) => setDefaultTubeId(e.target.value)} />
                </div>
                <div className="field">
                  <div className="label">Default Genotype</div>
                  <input className="input" value={defaultGenotype} onChange={(e) => setDefaultGenotype(e.target.value)} />
                </div>
                <div className="field">
                  <div className="label">Default Timepoint</div>
                  <input className="input" value={defaultTimepoint} onChange={(e) => setDefaultTimepoint(e.target.value)} />
                </div>
                <div className="field">
                  <div className="label">Depth Length (cm)</div>
                  <input className="input" type="number" min={0} step={0.5} value={defaultDepthLengthCm} onChange={(e) => setDefaultDepthLengthCm(Number(e.target.value))} />
                </div>
                <div className="field">
                  <div className="label">Camera</div>
                  <select className="select" value={cameraModel} onChange={(e) => setCameraModel(e.target.value as any)}>
                    <option value="ci600">CI-600 In-Situ Root Imager</option>
                  </select>
                </div>
                <div className="field">
                  <div className="label">DPI</div>
                  <input className="input" type="number" min={50} max={1200} step={50} value={ci600Dpi} onChange={(e) => setCi600Dpi(Number(e.target.value))} />
                </div>
                <div className="field">
                  <div className="label">Manual px→cm</div>
                  <select className="select" value={useManualPixelToCm ? "manual" : "auto"} onChange={(e) => setUseManualPixelToCm(e.target.value === "manual")}>
                    <option value="auto">Auto from DPI</option>
                    <option value="manual">Manual</option>
                  </select>
                </div>
                <div className="field">
                  <div className="label">px→cm (cm per pixel)</div>
                  <input className="input" type="number" step={1e-6} value={useManualPixelToCm ? manualPixelToCm : derivedPixelToCm} disabled={!useManualPixelToCm} onChange={(e) => setManualPixelToCm(Number(e.target.value))} />
                </div>
              </>
            ) : null}
          </div>
          <div className="hint">
            If enabled, this metadata is stored with each saved annotation so phenotyping + retraining can track camera/scaling and depth/timepoint.
          </div>
        </div>

        {analyzeError ? (
          <div style={{ marginTop: 10, color: "var(--danger)", fontSize: 13 }}>{analyzeError}</div>
        ) : null}
      </section>

      <section className="grid">
        <div className="panel">
          <div className="panelTitle">2) Review</div>
          <ProgressBar
            active={savingAll.active}
            value={savingAll.active && savingAll.total > 0 ? savingAll.done / savingAll.total : undefined}
            label={savingAll.active ? `Saving ${savingAll.done}/${savingAll.total}` : undefined}
          />
          {items.length === 0 ? (
            <div className="hint">Upload a batch to get started.</div>
          ) : (
            <div className="list">
              {items.map((it) => (
                <div
                  key={it.id}
                  className={`item ${it.id === selectedId ? "itemActive" : ""}`}
                  onClick={() => setSelectedId(it.id)}
                >
                  <div className="thumb">
                    <img src={it.originalDataUrl} alt={it.filename} />
                  </div>
                  <div className="meta">
                    <div className="filename">{it.filename}</div>
                    <div className={`status ${it.status === "error" ? "statusError" : ""}`}>
                      {it.status === "ready" ? "Ready" : null}
                      {it.status === "saving" ? "Saving…" : null}
                      {it.status === "saved" ? `Saved (${it.savedAnnotationId})` : null}
                      {it.status === "error" ? `Error: ${it.error}` : null}
                      {it.dirty && it.status !== "error" ? " · edited" : ""}
                    </div>
                    <div style={{ marginTop: 6 }}>
                      <ProgressBar active={it.status === "saving"} label={it.status === "saving" ? "Uploading…" : undefined} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {items.length > 0 ? (
            <div className="row" style={{ marginTop: 12 }}>
              <button className="btn btnPrimary" onClick={() => void saveAll()} disabled={!selectedItem || savingAll.active}>
                Save All
              </button>
              <div className="hint">Saves to `data/annotations/` via `POST /annotations`.</div>
            </div>
          ) : null}
        </div>

        <div className="panel editorWrap">
          <div className="panelTitle">3) Correct Mask</div>
          {!selectedItem ? (
            <div className="hint">Select an image from the list.</div>
          ) : selectedItem.status === "error" ? (
            <div className="hint">This image failed analysis; re-upload or check logs.</div>
          ) : (
            <>
              <div className="row">
                <button className={`btn ${tool === "add" ? "btnPrimary" : ""}`} onClick={() => setTool("add")}>
                  Add
                </button>
                <button className={`btn ${tool === "erase" ? "btnPrimary" : ""}`} onClick={() => setTool("erase")}>
                  Erase
                </button>

                <div className="field">
                  <div className="label">Brush (px)</div>
                  <input
                    className="input"
                    type="number"
                    min={2}
                    step={2}
                    value={brushSize}
                    onChange={(e) => setBrushSize(Number(e.target.value))}
                  />
                </div>

                <button className="btn" onClick={undo}>
                  Undo
                </button>

                <button
                  className="btn btnPrimary"
                  onClick={() => (selectedItem ? void saveItem(selectedItem.id) : undefined)}
                  disabled={selectedItem.status === "saving" || selectedItem.status === "saved"}
                >
                  Save This
                </button>
              </div>

              <div className="canvasFrame">
                <div className="canvasStage">
                  <img src={selectedItem.originalDataUrl} alt={selectedItem.filename} />
                  <canvas
                    ref={overlayCanvasRef}
                    onPointerDown={onPointerDown}
                    onPointerMove={onPointerMove}
                    onPointerUp={onPointerUp}
                    onPointerCancel={onPointerUp}
                    style={{ touchAction: "none" }}
                  />
                  <canvas ref={maskCanvasRef} style={{ display: "none" }} />
                </div>
              </div>

              <div className="hint">
                Paint roots (Add) or remove false positives (Erase). The saved mask is a binary PNG (white roots on black).
              </div>

              {selectedItem.metrics ? (
                <div className="panel" style={{ padding: 12 }}>
                  <div className="panelTitle">Metrics (from prediction)</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 8 }}>
                    <div className="hint">Root count: {selectedItem.metrics.root_count}</div>
                    <div className="hint">Avg diameter: {formatNumber(selectedItem.metrics.average_root_diameter)}</div>
                    <div className="hint">Total length: {formatNumber(selectedItem.metrics.total_root_length)}</div>
                    <div className="hint">Total area: {formatNumber(selectedItem.metrics.total_root_area)}</div>
                    <div className="hint">Total volume: {formatNumber(selectedItem.metrics.total_root_volume)}</div>
                  </div>
                </div>
              ) : null}

              {selectedItem.meta ? (
                <div className="panel" style={{ padding: 12 }}>
                  <div className="panelTitle">Capture Metadata</div>
                  <div className="row">
                    <div className="field">
                      <div className="label">Tube ID</div>
                      <input className="input" value={(selectedItem.meta as MiniMeta).tube_id ?? ""} onChange={(e) => patchSelectedMeta({ tube_id: e.target.value || undefined })} />
                    </div>
                    <div className="field">
                      <div className="label">Genotype</div>
                      <input className="input" value={(selectedItem.meta as MiniMeta).genotype ?? ""} onChange={(e) => patchSelectedMeta({ genotype: e.target.value || undefined })} />
                    </div>
                    <div className="field">
                      <div className="label">Depth</div>
                      <input className="input" type="number" min={0} step={1} value={(selectedItem.meta as MiniMeta).depth ?? ""} onChange={(e) => patchSelectedMeta({ depth: e.target.value ? Number(e.target.value) : undefined })} />
                    </div>
                    <div className="field">
                      <div className="label">Depth Length (cm)</div>
                      <input className="input" type="number" min={0} step={0.5} value={(selectedItem.meta as MiniMeta).depth_length_cm ?? ""} onChange={(e) => patchSelectedMeta({ depth_length_cm: e.target.value ? Number(e.target.value) : undefined })} />
                    </div>
                    <div className="field">
                      <div className="label">Timepoint</div>
                      <input className="input" value={(selectedItem.meta as MiniMeta).timepoint ?? ""} onChange={(e) => patchSelectedMeta({ timepoint: e.target.value || undefined })} />
                    </div>
                    <div className="field">
                      <div className="label">Session Label</div>
                      <input className="input" value={(selectedItem.meta as MiniMeta).session_label ?? ""} onChange={(e) => patchSelectedMeta({ session_label: e.target.value || undefined })} />
                    </div>
                  </div>
                  <div className="row">
                    <div className="hint">Camera: {(selectedItem.meta as MiniMeta).camera_model ?? "—"}</div>
                    <div className="hint">DPI: {(selectedItem.meta as MiniMeta).camera_dpi ?? "—"}</div>
                    <div className="hint">px→cm: {(selectedItem.meta as MiniMeta).pixel_to_cm ?? "—"}</div>
                    <button
                      className="btn"
                      onClick={() => {
                        const meta = buildDefaultMeta(selectedItem.filename);
                        if (!meta) return;
                        setItems((prev) => prev.map((it) => (it.id === selectedItem.id ? { ...it, meta } : it)));
                      }}
                    >
                      Apply Defaults
                    </button>
                  </div>
                </div>
              ) : null}
            </>
          )}
        </div>
      </section>
    </main>
  );
}
