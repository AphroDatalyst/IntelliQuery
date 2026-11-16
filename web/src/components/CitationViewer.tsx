import { useEffect, useRef, useState } from "react";

import { api } from "../api";
import { highlightStyle } from "../citations";
import type { Citation } from "../types";

interface Props {
  citation: Citation;
  onClose: () => void;
}

/** Shows the page a citation points at, with the cited region outlined. */
export function CitationViewer({ citation, onClose }: Props) {
  const [regionIndex, setRegionIndex] = useState(0);
  const [pageFailed, setPageFailed] = useState(false);
  const highlight = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setRegionIndex(0);
  }, [citation.chunk_id]);

  const region = citation.regions[Math.min(regionIndex, citation.regions.length - 1)];
  // A page render exists only when ingestion recorded the page's size.
  const hasPage = Boolean(region?.page_width) && !pageFailed;
  const style = region ? highlightStyle(region) : null;

  useEffect(() => {
    setPageFailed(false);
  }, [citation.chunk_id, region?.page]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <aside className="viewer" aria-label="Cited source">
      <header className="viewer-header">
        <span className="source-marker">{citation.marker}</span>
        <div className="viewer-title">
          <strong title={citation.filename}>{citation.filename}</strong>
          <span className="muted">
            {region ? `Page ${region.page}` : "No page recorded"}
            {citation.section_path ? ` · ${citation.section_path}` : ""}
          </span>
        </div>
        <button className="icon" aria-label="Close source" title="Close (Esc)" onClick={onClose}>
          ×
        </button>
      </header>

      {citation.regions.length > 1 && (
        <nav className="viewer-pages" aria-label="Pages of this passage">
          {citation.regions.map((r, index) => (
            <button
              key={r.page}
              className={index === regionIndex ? "tab active" : "tab"}
              onClick={() => setRegionIndex(index)}
            >
              Page {r.page}
            </button>
          ))}
        </nav>
      )}

      <div className="viewer-body">
        {citation.supported === false && (
          <p className="notice">The verification pass did not confirm this source.</p>
        )}

        {region && hasPage ? (
          <div className="page">
            <img
              key={`${citation.document_id}/${region.page}`}
              src={api.pageImageUrl(citation.document_id, region.page)}
              alt={`Page ${region.page} of ${citation.filename}`}
              onLoad={() =>
                highlight.current?.scrollIntoView({ block: "center", behavior: "smooth" })
              }
              onError={() => setPageFailed(true)}
            />
            {style && <div ref={highlight} className="highlight" style={style} />}
          </div>
        ) : (
          citation.has_image && (
            <img
              className="crop"
              src={api.chunkImageUrl(citation.chunk_id)}
              alt={`${citation.modality} from ${citation.filename}`}
            />
          )
        )}

        <h3>Cited {citation.modality === "text" ? "passage" : citation.modality}</h3>
        <pre className="quote">{citation.content}</pre>
      </div>
    </aside>
  );
}
