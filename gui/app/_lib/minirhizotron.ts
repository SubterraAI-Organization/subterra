export type ParsedMiniFilename = {
  field?: string;
  tube_id?: string;
  depth?: number;
};

function stripExtension(filename: string): string {
  return filename.replace(/\.[^.]+$/, "");
}

function normalizeField(value: string): string {
  return value.trim().toUpperCase();
}

function normalizeTubeId(value: string): string {
  const v = value.trim();
  const m = v.match(/^t?(\d+)$/i);
  if (m?.[1]) return `T${m[1]}`;
  return v.toUpperCase();
}

export function parseMiniFromFilename(filename: string): ParsedMiniFilename {
  const stem = stripExtension(filename);
  const tokens = stem.split(/[_-]+/g).map((t) => t.trim()).filter(Boolean);

  const out: ParsedMiniFilename = {};

  // Common CI-600 style: CS_T474_L007.PNG
  // - field/farm: CS
  // - tube_id: T474
  // - depth: L007 -> 7
  if (tokens.length >= 3) {
    const [a, b, c] = tokens;
    if (/^[A-Za-z]{2,10}$/.test(a) && /^T?\d{2,}$/.test(b) && /^[Ll]\d{1,4}$/.test(c)) {
      out.field = normalizeField(a);
      out.tube_id = normalizeTubeId(b);
      out.depth = Number.parseInt(c.slice(1), 10);
      return out;
    }
  }

  for (const token of tokens) {
    const lower = token.toLowerCase();

    if (!out.field && /^[A-Za-z]{2,10}$/.test(token)) {
      out.field = normalizeField(token);
      continue;
    }

    if (!out.tube_id) {
      // tube01 / tube_01 / T474
      const tube =
        lower.match(/^(?:tube|t)(\d{1,6})$/i) ??
        lower.match(/^tube[_-]?(\d{1,6})$/i);
      if (tube?.[1]) {
        out.tube_id = `T${tube[1]}`;
        continue;
      }
    }

    if (out.depth === undefined) {
      // depth7 / d7 / L007
      const depth =
        lower.match(/^depth(\d{1,4})$/i) ??
        lower.match(/^d(\d{1,4})$/i) ??
        lower.match(/^l(\d{1,4})$/i);
      if (depth?.[1]) {
        out.depth = Number.parseInt(depth[1], 10);
        continue;
      }
    }

  }

  return out;
}
