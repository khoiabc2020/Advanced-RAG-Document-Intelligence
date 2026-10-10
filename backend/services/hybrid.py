"""Hybrid search: BM25-style FTS lane + dense lane, fused with RRF.

Only Postgres has a native FTS lane today (uses `content_tsv` generated
column + GIN index created in migration v5). On SQLite the FTS lane
returns an empty list, so hybrid retrieval transparently degrades to
the dense lane. Fusion is Reciprocal Rank Fusion (Cormack et al. 2009),
the industry default for multi-signal retrieval:

    score(d) = sum_i  weight_i / (k + rank_i(d))
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from backend.database import fetch_all
from backend.services.vectorstore import RetrievedChunk
from backend.settings import settings

_VI_ORTHOGRAPHY = {
    "kì": "kỳ", "kĩ": "kỹ", "kí": "ký", "kỉ": "kỷ", "kị": "kỵ",
    "lí": "lý", "lì": "lỳ", "lỉ": "lỷ", "lĩ": "lỹ", "lị": "lỵ",
    "mĩ": "mỹ", "mí": "mý", "mì": "mỳ", "mỉ": "mỷ", "mị": "mỵ",
    "sĩ": "sỹ", "sí": "sý", "sì": "sỳ", "sỉ": "sỷ", "sị": "sỵ",
    "tỉ": "tỷ", "tí": "tý", "tì": "tỳ", "tĩ": "tỹ", "tị": "tỵ",
    "quĩ": "quỹ", "quí": "quý", "quì": "quỳ", "quỉ": "quỷ", "quị": "quỵ",
}

_STOPWORDS = {
    "theo", "tài", "liệu", "tại", "nào", "ta", "về", "đó", "và", "cho",
    "biết", "này", "của", "nước", "trong", "là", "gì", "được", "ra", "đầy",
    "đủ", "nêu", "chính", "xác", "các", "những",
}


def _norm(t: str) -> str:
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFC", t.lower())).strip()
    for old, new in _VI_ORTHOGRAPHY.items():
        s = re.sub(r"\b" + old + r"\b", new, s)
    return s


def _score_rows_for_text(rows: list[dict], query_text: str) -> list[tuple[float, dict]]:
    norm_text = _norm(query_text)
    raw_quotes = [
        _norm(q)
        for q in re.findall(r"['\"‘“]([^'\"’”]{3,})['\"’”]", query_text)
        if _norm(q)
    ]
    exact_quotes: list[str] = list(raw_quotes)
    sub_quotes: list[str] = []
    for rq in raw_quotes:
        words = rq.split()
        if len(words) > 3:
            for n in range(3, min(len(words), 6)):
                for i in range(len(words) - n + 1):
                    sub_quotes.append(" ".join(words[i : i + n]))

    words = re.findall(r"[\w\u00C0-\u1EF9]+", norm_text)
    key_phrases: list[str] = []
    for n in (3, 4, 5):
        for i in range(len(words) - n + 1):
            window = words[i : i + n]
            content_words = [w for w in window if w not in _STOPWORDS]
            if len(content_words) >= 2 and window[0] not in _STOPWORDS and window[-1] not in _STOPWORDS:
                key_phrases.append(" ".join(window))
    key_phrases = list(dict.fromkeys(key_phrases))

    tokens = [t for t in words if len(t) > 2 and t not in _STOPWORDS]
    if not tokens and not exact_quotes and not key_phrases:
        tokens = [t for t in words if len(t) > 2]

    scored: list[tuple[float, dict]] = []
    for row in rows:
        content_norm = _norm(row.get("content") or "")
        score = 0.0
        exact_matched = False
        for q in exact_quotes:
            if q in content_norm:
                score += 100.0
                exact_matched = True
        for sq in sub_quotes:
            if sq in content_norm:
                score += 30.0
                exact_matched = True
        for kp in key_phrases:
            if kp in content_norm:
                score += 20.0 * len(kp.split())
                exact_matched = True

        matched_tokens = 0
        for t in tokens:
            cnt = content_norm.count(t)
            if cnt > 0:
                matched_tokens += 1
                score += 1.0 + min(1.0, 0.2 * cnt)
        if score > 0:
            coverage = matched_tokens / len(tokens) if tokens else 0.0
            score += coverage * 15.0
            if exact_matched:
                score += 50.0
            scored.append((score, row))

    scored.sort(key=lambda x: -x[0])
    return scored


async def fts_search(
    document_ids: list[str],
    owner_id: str,
    query: str,
    top_k: int,
    sub_queries: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Lexical lane with exact-quote matching, multi-clause splitting, and token coverage."""
    if not document_ids or not query.strip():
        return []

    placeholders = ",".join(["?"] * len(document_ids))
    sql = f"""
        SELECT id, document_id, content, page_number, chunk_index
        FROM document_chunks
        WHERE owner_id = ?
          AND document_id IN ({placeholders})
    """
    rows = await fetch_all(sql, (owner_id, *document_ids))
    if not rows:
        return []

    # Use explicit sub_queries (from LLM decomposition) if available, otherwise regex splitting
    active_sub_clauses = sub_queries if (sub_queries and len(sub_queries) > 0) else None
    if not active_sub_clauses:
        split_patterns = r"[?.;!]\s*|\s+(?:và cho biết|đồng thời|và nêu|cho biết|và xác định)\s+"
        parts = [c.strip() for c in re.split(split_patterns, query) if len(c.strip()) > 8]
        if len(parts) > 1:
            active_sub_clauses = parts

    if active_sub_clauses:
        # Multi-clause / decomposed fusion: score full query + each sub-query to prevent sub-question starvation
        sub_lanes: list[tuple[list[RetrievedChunk], float]] = []
        full_scored = _score_rows_for_text(rows, query)
        candidate_k = max(top_k, 25)
        if full_scored:
            sub_lanes.append((
                [
                    RetrievedChunk(
                        chunk_id=row["id"],
                        document_id=row["document_id"],
                        excerpt=row["content"],
                        score=min(1.0, round(score / 50.0, 4)),
                        page_number=row.get("page_number"),
                        chunk_index=row.get("chunk_index"),
                    )
                    for score, row in full_scored[:candidate_k]
                ],
                0.8,
            ))
        for clause in active_sub_clauses:
            clause_scored = _score_rows_for_text(rows, clause)
            if clause_scored:
                sub_lanes.append((
                    [
                        RetrievedChunk(
                            chunk_id=row["id"],
                            document_id=row["document_id"],
                            excerpt=row["content"],
                            score=min(1.0, round(score / 50.0, 4)),
                            page_number=row.get("page_number"),
                            chunk_index=row.get("chunk_index"),
                        )
                        for score, row in clause_scored[:candidate_k]
                    ],
                    1.2,
                ))
        return reciprocal_rank_fusion(sub_lanes, top_k=top_k)

    scored = _score_rows_for_text(rows, query)
    return [
        RetrievedChunk(
            chunk_id=row["id"],
            document_id=row["document_id"],
            excerpt=row["content"],
            score=min(1.0, round(score / 50.0, 4)),
            page_number=row.get("page_number"),
            chunk_index=row.get("chunk_index"),
        )
        for score, row in scored[:top_k]
    ]


def reciprocal_rank_fusion(
    lanes: list[tuple[list[RetrievedChunk], float]],
    *,
    top_k: int,
    rrf_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked lists via weighted RRF.

    `lanes` is a list of `(ranked_chunks, weight)` pairs. Chunks are
    identified by `chunk_id`; the first occurrence is used as the
    representative payload, and the fused RRF score is normalized to [0, 1].
    """
    k = rrf_k if rrf_k is not None else settings.hybrid_rrf_k
    scores: dict[str, float] = defaultdict(float)
    representatives: dict[str, RetrievedChunk] = {}

    for ranked, weight in lanes:
        for rank_index, chunk in enumerate(ranked):
            scores[chunk.chunk_id] += weight / (k + rank_index + 1)
            representatives.setdefault(chunk.chunk_id, chunk)

    max_possible = sum(weight for _, weight in lanes) / (k + 1)
    fused = [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            excerpt=chunk.excerpt,
            score=round(min(1.0, scores[chunk.chunk_id] / max_possible), 4) if max_possible > 0 else 0.0,
            page_number=chunk.page_number,
        )
        for chunk in representatives.values()
    ]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused[:top_k]
