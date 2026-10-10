"""Window & Section Expansion service (Small-to-Big Retrieval).

Retrieves adjacent contiguous chunks (chunk_index - 1, +1, +2) for high-ranked
seed chunks to prevent semantic boundary severing (e.g. headings separated from
their bullet points or multi-page narratives).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from backend.database import fetch_all
from backend.services.vectorstore import RetrievedChunk

logger = logging.getLogger("ragapp.window_expansion")

_CONTINUATION_PATTERNS = re.compile(
    r"(?::\s*$|như sau\s*:?$|gồm(?:\s+có)?\s*:?$|nội dung(?:\s+chính)?\s*:?$|cụ thể\s*:?$|sau đây\s*:?$)",
    re.IGNORECASE,
)


def _needs_forward_expansion(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if _CONTINUATION_PATTERNS.search(stripped):
        return True
    # If chunk ends without standard closing punctuation (., !, ?, ", ”), it was likely clipped
    if stripped[-1] not in (".", "!", "?", '"', "”", "'", "’", ";"):
        return True
    return False


async def expand_chunk_windows(
    chunks: list[RetrievedChunk],
    *,
    owner_id: str,
    max_total_chunks: int = 28,
    force_expand: bool = False,
) -> tuple[list[RetrievedChunk], dict[str, Any]]:
    """Expand retrieved chunks with adjacent context chunks for continuity.

    Keeps contiguous sections intact without discarding expanded sibling chunks.
    """
    if not chunks:
        return chunks, {"expanded_added": 0}

    # Map existing chunks by (doc_id, chunk_index)
    existing: dict[tuple[str, int], RetrievedChunk] = {}
    doc_chunk_scores: dict[tuple[str, int], float] = {}

    for c in chunks:
        doc_id = c.document_id
        idx: int | None = getattr(c, "chunk_index", None)
        if idx is None:
            parts = c.chunk_id.split(":")
            if len(parts) > 1 and parts[-1].isdigit():
                idx = int(parts[-1])
        if idx is not None:
            existing[(doc_id, idx)] = c
            doc_chunk_scores[(doc_id, idx)] = c.score

    # Determine which adjacent indices to fetch for each seed
    missing_indices: dict[str, set[int]] = {}
    parent_score_map: dict[tuple[str, int], float] = {}

    for (doc_id, idx), c in existing.items():
        # Generic multi-chunk expansion for all documents
        needed_forward = 2 if force_expand else (1 if _needs_forward_expansion(c.excerpt) else 0)
        needed_backward = 1 if force_expand else 0

        for step in range(1, needed_forward + 1):
            target_idx = idx + step
            if (doc_id, target_idx) not in existing:
                missing_indices.setdefault(doc_id, set()).add(target_idx)
                parent_score_map[(doc_id, target_idx)] = max(
                    parent_score_map.get((doc_id, target_idx), 0.0),
                    c.score,
                )

        if needed_backward and idx > 0:
            target_idx = idx - 1
            if (doc_id, target_idx) not in existing:
                missing_indices.setdefault(doc_id, set()).add(target_idx)
                parent_score_map[(doc_id, target_idx)] = max(
                    parent_score_map.get((doc_id, target_idx), 0.0),
                    c.score,
                )

    total_missing = sum(len(indices) for indices in missing_indices.values())
    added_chunks: list[RetrievedChunk] = []

    if total_missing:
        for doc_id, indices in missing_indices.items():
            if not indices:
                continue
            sorted_indices = sorted(indices)
            placeholders = ",".join(["?"] * len(sorted_indices))
            sql = f"""
                SELECT id, document_id, content, page_number, chunk_index
                FROM document_chunks
                WHERE owner_id = ? AND document_id = ? AND chunk_index IN ({placeholders})
                ORDER BY chunk_index ASC
            """
            try:
                rows = await fetch_all(sql, (owner_id, doc_id, *sorted_indices))
                for row in rows:
                    c_idx = row["chunk_index"]
                    inherited_score = parent_score_map.get((doc_id, c_idx), 0.80)
                    added_chunk = RetrievedChunk(
                        chunk_id=row["id"],
                        document_id=row["document_id"],
                        excerpt=row["content"],
                        score=round(inherited_score, 4),
                        page_number=row.get("page_number"),
                        chunk_index=c_idx,
                    )
                    added_chunks.append(added_chunk)
                    existing[(doc_id, c_idx)] = added_chunk
            except Exception as exc:
                logger.warning("window_expansion_fetch_failed doc=%s err=%s", doc_id, exc)

    # Combine all chunks
    combined: list[RetrievedChunk] = list(existing.values())

    # Sort in sequential reading order by document_id and chunk_index
    def _reading_order_key(c: RetrievedChunk):
        parts = c.chunk_id.split(":")
        idx = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else (c.chunk_index or 0)
        return (c.document_id, idx)

    combined.sort(key=_reading_order_key)

    # If exceeding max_total_chunks, cap carefully by pruning isolated low-score singletons first
    if len(combined) > max_total_chunks:
        # Sort by score descending to get top candidates, but ensure adjacent neighbors stay together
        combined.sort(key=lambda x: -x.score)
        combined = combined[:max_total_chunks]
        combined.sort(key=_reading_order_key)

    stats = {
        "expanded_added": len(added_chunks),
        "total_after_expansion": len(combined),
    }
    return combined, stats
