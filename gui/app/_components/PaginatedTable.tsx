"use client";

import React, { useEffect, useMemo, useState } from "react";

export type SortDir = "asc" | "desc";

export type PaginatedTableColumn<T> = {
  id: string;
  header: React.ReactNode;
  cell: (row: T) => React.ReactNode;
  sortValue?: (row: T) => string | number | null | undefined;
  className?: string;
};

type Props<T> = {
  rows: T[];
  columns: Array<PaginatedTableColumn<T>>;
  getRowKey: (row: T, index: number) => string;
  emptyLabel?: string;
  initialPageSize?: number;
  pageSizeOptions?: number[];
  searchableText?: (row: T) => string;
  initialQuery?: string;
  initialSort?: { columnId: string; dir: SortDir };
  isRowActive?: (row: T) => boolean;
  footerLeft?: React.ReactNode;
};

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function asSortable(v: string | number | null | undefined): string | number | null {
  if (v == null) return null;
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  const s = String(v).trim();
  return s.length ? s.toLowerCase() : null;
}

function compareNullable(a: string | number | null, b: string | number | null) {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b));
}

export function PaginatedTable<T>({
  rows,
  columns,
  getRowKey,
  emptyLabel = "No rows.",
  initialPageSize = 25,
  pageSizeOptions = [10, 25, 50, 100],
  searchableText,
  initialQuery = "",
  initialSort,
  isRowActive,
  footerLeft
}: Props<T>) {
  const [query, setQuery] = useState(initialQuery);
  const [pageSize, setPageSize] = useState(() => (pageSizeOptions.includes(initialPageSize) ? initialPageSize : pageSizeOptions[0] ?? 25));
  const [pageIndex, setPageIndex] = useState(0);
  const [sort, setSort] = useState<{ columnId: string; dir: SortDir } | null>(initialSort ?? null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q || !searchableText) return rows;
    return rows.filter((r) => searchableText(r).toLowerCase().includes(q));
  }, [rows, query, searchableText]);

  const sorted = useMemo(() => {
    if (!sort) return filtered;
    const col = columns.find((c) => c.id === sort.columnId);
    if (!col?.sortValue) return filtered;

    const dirMul = sort.dir === "asc" ? 1 : -1;
    return filtered
      .map((row, idx) => ({ row, idx, v: asSortable(col.sortValue?.(row)) }))
      .sort((a, b) => {
        const cmp = compareNullable(a.v, b.v);
        if (cmp !== 0) return cmp * dirMul;
        return a.idx - b.idx;
      })
      .map((x) => x.row);
  }, [filtered, sort, columns]);

  const total = sorted.length;
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const safePageIndex = clamp(pageIndex, 0, totalPages - 1);

  useEffect(() => {
    if (safePageIndex !== pageIndex) setPageIndex(safePageIndex);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [safePageIndex]);

  useEffect(() => {
    setPageIndex(0);
  }, [query, pageSize, sort?.columnId, sort?.dir]);

  const start = safePageIndex * pageSize;
  const end = Math.min(total, start + pageSize);
  const pageRows = useMemo(() => sorted.slice(start, end), [sorted, start, end]);

  function toggleSort(colId: string) {
    const col = columns.find((c) => c.id === colId);
    if (!col?.sortValue) return;
    setSort((prev) => {
      if (!prev || prev.columnId !== colId) return { columnId: colId, dir: "asc" };
      return prev.dir === "asc" ? { columnId: colId, dir: "desc" } : null;
    });
  }

  const canSearch = Boolean(searchableText);
  const showingLabel = total === 0 ? "0" : `${start + 1}–${end}`;

  return (
    <div style={{ display: "grid", gap: 10 }}>
      <div className="tableToolbar">
        <div className="tableToolbarLeft">
          {canSearch ? (
            <input
              className="input tableSearch"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search…"
              aria-label="Search rows"
            />
          ) : null}
          <div className="hint">
            Showing {showingLabel} of {total}
          </div>
        </div>
        <div className="tableToolbarRight">
          <div className="field" style={{ minWidth: 0 }}>
            <div className="label">Rows</div>
            <select className="select" value={pageSize} onChange={(e) => setPageSize(Number(e.target.value))}>
              {pageSizeOptions.map((n) => (
                <option key={n} value={n}>
                  {n} / page
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {total === 0 ? (
        <div className="hint">{emptyLabel}</div>
      ) : (
        <div className="tableWrap">
          <table className="dataTable">
            <thead>
              <tr>
                {columns.map((c) => {
                  const sortable = Boolean(c.sortValue);
                  const active = sort?.columnId === c.id;
                  const indicator = active ? (sort?.dir === "asc" ? " ▲" : " ▼") : "";
                  return (
                    <th key={c.id} className={c.className}>
                      {sortable ? (
                        <button className="tableSortBtn" onClick={() => toggleSort(c.id)} type="button">
                          {c.header}
                          <span aria-hidden="true">{indicator}</span>
                        </button>
                      ) : (
                        c.header
                      )}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {pageRows.map((row, idx) => {
                const key = getRowKey(row, start + idx);
                const active = isRowActive?.(row) ?? false;
                return (
                  <tr key={key} data-active={active ? "true" : "false"}>
                    {columns.map((c) => (
                      <td key={c.id} className={c.className}>
                        {c.cell(row)}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="tableFooter">
        <div className="tableFooterLeft">{footerLeft}</div>
        <div className="pager">
          <button className="btn" style={{ padding: "6px 8px" }} onClick={() => setPageIndex(0)} disabled={safePageIndex === 0}>
            First
          </button>
          <button
            className="btn"
            style={{ padding: "6px 8px" }}
            onClick={() => setPageIndex((p) => Math.max(0, p - 1))}
            disabled={safePageIndex === 0}
          >
            Prev
          </button>
          <div className="hint" style={{ minWidth: 120, textAlign: "center" }}>
            Page {safePageIndex + 1} / {totalPages}
          </div>
          <button
            className="btn"
            style={{ padding: "6px 8px" }}
            onClick={() => setPageIndex((p) => Math.min(totalPages - 1, p + 1))}
            disabled={safePageIndex >= totalPages - 1}
          >
            Next
          </button>
          <button
            className="btn"
            style={{ padding: "6px 8px" }}
            onClick={() => setPageIndex(totalPages - 1)}
            disabled={safePageIndex >= totalPages - 1}
          >
            Last
          </button>
        </div>
      </div>
    </div>
  );
}

