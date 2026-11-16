import type { Box, Region } from "./types";

export type Segment = { kind: "text"; text: string } | { kind: "cite"; marker: number };

const MARKER = /\[(\d+)\]/g;

/**
 * Split an answer into plain text and citation markers.
 *
 * Only markers that match a citation of the answer become links; any other
 * bracketed number (a quoted "[3]" from a document, say) stays as text.
 */
export function splitAnswer(text: string, markers: ReadonlySet<number>): Segment[] {
  const segments: Segment[] = [];
  let pending = "";
  let cursor = 0;
  for (const match of text.matchAll(MARKER)) {
    const marker = Number(match[1]);
    if (!markers.has(marker)) continue;
    // The space before a marker belongs to the marker, not the sentence.
    pending += text.slice(cursor, match.index).replace(/ $/, "");
    if (pending) segments.push({ kind: "text", text: pending });
    pending = "";
    segments.push({ kind: "cite", marker });
    cursor = match.index + match[0].length;
  }
  pending += text.slice(cursor);
  if (pending) segments.push({ kind: "text", text: pending });
  return segments;
}

export interface Highlight {
  left: string;
  top: string;
  width: string;
  height: string;
}

/** Position of a cited box as percentages of its page, so it scales with the image. */
export function highlightStyle(region: Region): Highlight | null {
  const { box, page_width: width, page_height: height } = region;
  if (!box || !width || !height) return null;
  const clamp = (value: number) => Math.min(100, Math.max(0, value));
  const { x0, y0, x1, y1 }: Box = box;
  const left = clamp((Math.min(x0, x1) / width) * 100);
  const top = clamp((Math.min(y0, y1) / height) * 100);
  const right = clamp((Math.max(x0, x1) / width) * 100);
  const bottom = clamp((Math.max(y0, y1) / height) * 100);
  if (right <= left || bottom <= top) return null;
  return {
    left: `${left}%`,
    top: `${top}%`,
    width: `${right - left}%`,
    height: `${bottom - top}%`,
  };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
