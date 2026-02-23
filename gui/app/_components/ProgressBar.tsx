"use client";

import React from "react";

export function ProgressBar({
  active,
  value,
  label,
  reserveSpace
}: {
  active: boolean;
  value?: number;
  label?: string;
  reserveSpace?: boolean;
}) {
  const visible = active || value != null;
  if (!visible && !reserveSpace) return null;

  const clamped = value == null ? null : Math.max(0, Math.min(1, value));
  return (
    <div className="progressWrap" aria-label={label} style={!visible ? { opacity: 0, pointerEvents: "none" } : undefined}>
      <div className={`progressTrack ${active && clamped == null ? "progressIndeterminate" : ""}`}>
        {clamped != null ? <div className="progressFill" style={{ width: `${clamped * 100}%` }} /> : null}
      </div>
      {label ? <div className="progressLabel">{label}</div> : null}
    </div>
  );
}
