"use client";

import React, { useEffect, useMemo, useState } from "react";
import { ProgressBar } from "../_components/ProgressBar";
import { PaginatedTable, type PaginatedTableColumn } from "../_components/PaginatedTable";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001";

type ApiKeyItem = {
  key_id: string;
  name: string;
  created_at: string;
  last_used_at?: string | null;
  revoked: boolean;
};

type ApiKeyListResponse = { keys: ApiKeyItem[] };
type ApiKeyCreateResponse = { key_id: string; name: string; created_at: string; api_key: string };

export default function Page() {
  const [adminToken, setAdminToken] = useState<string>("");
  const [name, setName] = useState<string>("automation");
  const [creating, setCreating] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [keys, setKeys] = useState<ApiKeyItem[]>([]);
  const [created, setCreated] = useState<ApiKeyCreateResponse | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem("subterra_admin_token");
    if (saved) setAdminToken(saved);
  }, []);

  useEffect(() => {
    localStorage.setItem("subterra_admin_token", adminToken);
  }, [adminToken]);

  async function refresh() {
    if (!adminToken) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api-keys`, {
        headers: { "X-Admin-Token": adminToken }
      });
      if (!res.ok) throw new Error(await res.text());
      const data = (await res.json()) as ApiKeyListResponse;
      setKeys(data.keys);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, [adminToken]);

  async function createKey() {
    setCreated(null);
    setCreating(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api-keys`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Admin-Token": adminToken },
        body: JSON.stringify({ name })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = (await res.json()) as ApiKeyCreateResponse;
      setCreated(data);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCreating(false);
    }
  }

  async function revokeKey(keyId: string) {
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api-keys/${encodeURIComponent(keyId)}/revoke`, {
        method: "POST",
        headers: { "X-Admin-Token": adminToken }
      });
      if (!res.ok) throw new Error(await res.text());
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const curlIngestExample = useMemo(() => {
    const apiKey = created?.api_key ? created.api_key : "<YOUR_API_KEY>";
    return `curl -X POST "${API_BASE_URL}/ingest/analysis" \\\n` +
      `  -H "Content-Type: application/json" \\\n` +
      `  -H "X-API-Key: ${apiKey}" \\\n` +
      `  -d '{\n` +
      `    "filename": "sample1.png",\n` +
      `    "model_type": "unet",\n` +
      `    "model_version": "unet_v0003",\n` +
      `    "threshold_area": 50,\n` +
      `    "scaling_factor": 1.0,\n` +
      `    "confidence_threshold": 0.0,\n` +
      `    "root_count": 5,\n` +
      `    "average_root_diameter": 2.34,\n` +
      `    "total_root_length": 156.78,\n` +
      `    "total_root_area": 89.45,\n` +
      `    "total_root_volume": 23.67,\n` +
      `    "extra": {"source": "external_pipeline"}\n` +
      `  }'`;
  }, [created?.api_key]);

  const keyColumns: Array<PaginatedTableColumn<ApiKeyItem>> = [
    { id: "name", header: "name", sortValue: (k) => k.name, cell: (k) => k.name },
    { id: "key_id", header: "key_id", sortValue: (k) => k.key_id, cell: (k) => k.key_id },
    { id: "created_at", header: "created_at", sortValue: (k) => k.created_at, cell: (k) => k.created_at },
    { id: "last_used_at", header: "last_used_at", sortValue: (k) => k.last_used_at ?? "", cell: (k) => k.last_used_at ?? "—" },
    { id: "revoked", header: "revoked", sortValue: (k) => (k.revoked ? 1 : 0), cell: (k) => (k.revoked ? "yes" : "no") },
    {
      id: "actions",
      header: "actions",
      cell: (k) => (
        <button className="btn btnDanger" style={{ padding: "6px 8px" }} disabled={k.revoked} onClick={() => void revokeKey(k.key_id)}>
          Revoke
        </button>
      )
    }
  ];

  return (
    <main style={{ display: "grid", gap: 16 }}>
      <header className="panel">
        <h1 style={{ margin: 0, fontSize: 18 }}>API (Keys + Ingestion)</h1>
        <p style={{ margin: "6px 0 0", color: "var(--muted)", fontSize: 13 }}>
          Create API keys for automated ingestion pipelines (stores into Postgres).
        </p>
        <div className="hint" style={{ marginTop: 8 }}>
          API: {API_BASE_URL}
        </div>
        {error ? <div style={{ marginTop: 10, color: "var(--danger)", fontSize: 13 }}>{error}</div> : null}
      </header>

      <section className="grid">
        <div className="panel">
          <div className="panelTitle">Admin</div>
          <div className="hint">
            To manage keys, set `SUBTERRA_ADMIN_TOKEN` on the API container and paste it here.
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <div className="field">
              <div className="label">Admin Token</div>
              <input className="input" value={adminToken} onChange={(e) => setAdminToken(e.target.value)} placeholder="SUBTERRA_ADMIN_TOKEN" />
            </div>
            <button className="btn" onClick={() => void refresh()} disabled={!adminToken || loading}>
              Refresh
            </button>
          </div>
          <div style={{ marginTop: 10 }}>
            <ProgressBar active={loading} label={loading ? "Loading keys…" : undefined} reserveSpace />
          </div>
        </div>

        <div className="panel">
          <div className="panelTitle">Create API Key</div>
          <div className="row">
            <div className="field">
              <div className="label">Name</div>
              <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <button className="btn btnPrimary" onClick={() => void createKey()} disabled={!adminToken || creating}>
              Create
            </button>
          </div>
          <div style={{ marginTop: 10 }}>
            <ProgressBar active={creating} label={creating ? "Creating…" : undefined} reserveSpace />
          </div>
          {created ? (
            <div className="panel" style={{ marginTop: 12, padding: 12 }}>
              <div className="panelTitle">New Key (shown once)</div>
              <div className="hint">Key ID: {created.key_id}</div>
              <div className="hint" style={{ marginTop: 6, wordBreak: "break-all" }}>
                API Key: {created.api_key}
              </div>
            </div>
          ) : null}
        </div>
      </section>

      <section className="panel">
        <div className="panelTitle">Ingest Example</div>
        <div className="hint">Use `X-API-Key` (or `Authorization: Bearer ...`) to ingest externally computed phenotypes:</div>
        <pre
          style={{
            margin: "10px 0 0",
            padding: 12,
            borderRadius: 12,
            border: "1px solid var(--border)",
            background: "rgba(0,0,0,0.2)",
            overflow: "auto",
            fontSize: 12,
            color: "rgba(233, 237, 255, 0.95)"
          }}
        >
          {curlIngestExample}
        </pre>
      </section>

      <section className="panel">
        <div className="panelTitle">Existing Keys</div>
        {!adminToken ? (
          <div className="hint">Enter admin token to list keys.</div>
        ) : keys.length === 0 ? (
          <div className="hint">No keys yet.</div>
        ) : (
          <PaginatedTable
            rows={keys}
            columns={keyColumns}
            getRowKey={(k) => k.key_id}
            initialSort={{ columnId: "created_at", dir: "desc" }}
            initialPageSize={25}
            searchableText={(k) => `${k.name} ${k.key_id} ${k.created_at} ${k.last_used_at ?? ""} ${k.revoked ? "revoked" : ""}`}
            emptyLabel="No keys."
          />
        )}
      </section>
    </main>
  );
}
