"""Retrieval pipeline orchestrator.

Stages (each independently flag-gated):

    query -> [embed] -> dense lane
                     -> (optional) FTS lane
                     -> RRF fusion
                     -> (optional) cross-encoder rerank
                     -> final top_k chunks

Every stage opens a child span under the provided trace so latencies
and counts are observable with or without Langfuse. The orchestrator is
deliberately stateless; callers own embedding and LLM generation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from backend.services import graph_retrieval, hybrid, mmr as mmr_mod, reranker, tracing
from backend.services.window_expansion import expand_chunk_windows
from backend.services.vectorstore import (
    RetrievedChunk,
    query_similar,
    query_similar_multi,
)
from backend.settings import settings


@dataclass(slots=True)
class RetrievalRequest:
    """Inputs for a single retrieval call: embeddings, scope, and optional stage overrides."""

    query: str
    document_ids: list[str]
    owner_id: str
    query_embedding: list[float]
    top_k: int
    min_score: float = 0.0
    workspace_id: str | None = None
    hybrid_enabled: bool | None = None
    reranker_enabled: bool | None = None
    mmr_enabled: bool | None = None
    # Optional extra dense lanes (e.g. from HyDE / multi-query / decomposition).
    extra_query_embeddings: list[tuple[str, list[float]]] = field(default_factory=list)
    # Optional decomposed sub-queries for targeted multi-hop / comparison retrieval
    sub_queries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalResult:
    """Final ranked chunks plus a ``stages`` dict for observability (counts, flags, timings)."""

    chunks: list[RetrievedChunk]
    stages: dict[str, Any] = field(default_factory=dict)


async def _dense_search(
    document_ids: list[str],
    owner_id: str,
    embedding: list[float],
    k: int,
    min_score: float,
    workspace_id: str | None = None,
) -> list[RetrievedChunk]:
    if len(document_ids) == 1:
        return await query_similar(document_ids[0], owner_id, embedding, k, min_score, workspace_id)
    return await query_similar_multi(document_ids, owner_id, embedding, k, min_score, workspace_id)


async def retrieve(req: RetrievalRequest, *, trace_span: tracing._Span | None = None) -> RetrievalResult:
    """Run dense (+ optional hybrid, rerank, MMR) retrieval and return ranked chunks."""
    hybrid_on = (
        settings.retrieval_hybrid_enabled if req.hybrid_enabled is None else req.hybrid_enabled
    )
    rerank_on = (
        settings.retrieval_reranker_enabled
        if req.reranker_enabled is None
        else req.reranker_enabled
    )
    mmr_on = (
        settings.retrieval_mmr_enabled if req.mmr_enabled is None else req.mmr_enabled
    )

    # Over-fetch when we're going to rerank or MMR-diversify or hybrid-fuse so downstream
    # stages have a meaningful candidate pool to work with.
    overfetch_factor = 1
    if rerank_on:
        overfetch_factor = max(overfetch_factor, settings.reranker_overfetch_factor)
    if mmr_on:
        overfetch_factor = max(overfetch_factor, 3)
    if hybrid_on:
        overfetch_factor = max(overfetch_factor, 3)
    dense_k = max(req.top_k, req.top_k * overfetch_factor)
    if req.sub_queries or req.extra_query_embeddings:
        dense_k = max(dense_k, 25)

    stages: dict[str, Any] = {
        "hybrid_enabled": hybrid_on,
        "reranker_enabled": rerank_on,
        "mmr_enabled": mmr_on,
        "dense_k": dense_k,
        "requested_top_k": req.top_k,
        "extra_query_lanes": len(req.extra_query_embeddings),
        "sub_queries": req.sub_queries,
    }

    # --- Primary dense lane ---
    with tracing.span(trace_span, "dense_retrieval", k=dense_k) as dense_span:
        dense_hits = await _dense_search(
            req.document_ids, req.owner_id, req.query_embedding, dense_k, req.min_score, req.workspace_id
        )
        dense_span.update(hits=len(dense_hits))
    stages["dense_hits"] = len(dense_hits)

    # --- Extra dense lanes (HyDE / multi-query / decomposition) ---
    extra_lanes: list[tuple[list[RetrievedChunk], float]] = []
    if req.extra_query_embeddings:
        with tracing.span(
            trace_span,
            "extra_query_lanes",
            lanes=len(req.extra_query_embeddings),
        ) as xq_span:
            tasks = [
                _dense_search(
                    req.document_ids, req.owner_id, embedding, dense_k, req.min_score, req.workspace_id
                )
                for _label, embedding in req.extra_query_embeddings
            ]
            results = await asyncio.gather(*tasks)

            total_hits = 0
            for (label, _), hits in zip(req.extra_query_embeddings, results):
                # Decomposed sub-queries get full vector weight so sub-entities are not starved
                w = settings.hybrid_vector_weight if label == "decomposition" else (settings.hybrid_vector_weight * 0.5)
                extra_lanes.append((hits, w))
                total_hits += len(hits)
            xq_span.update(hits=total_hits)
        stages["extra_lane_hits"] = sum(len(h) for h, _ in extra_lanes)

    # --- Optional graph-traversal lane (Phase 3.5) ---
    graph_hits: list[RetrievedChunk] = []
    graph_on = (
        settings.retrieval_graph_traversal_enabled
        and settings.retrieval_graph_enabled
    )
    if graph_on:
        with tracing.span(trace_span, "graph_retrieval", k=dense_k) as graph_span:
            graph_result = await graph_retrieval.graph_lane_search(
                owner_id=req.owner_id,
                document_ids=req.document_ids,
                question=req.query,
                top_k=dense_k,
            )
            graph_hits = graph_result.chunks
            graph_span.update(hits=len(graph_hits), seeds=len(graph_result.stats.get("seeds", []) or []))
        stages["graph_hits"] = len(graph_hits)
        stages["graph_stats"] = graph_result.stats

    # --- Optional FTS lane + RRF fusion ---
    candidates: list[RetrievedChunk]
    if hybrid_on or extra_lanes or graph_hits:
        fusion_inputs: list[tuple[list[RetrievedChunk], float]] = [
            (dense_hits, settings.hybrid_vector_weight)
        ]
        if hybrid_on:
            with tracing.span(trace_span, "fts_retrieval", k=dense_k) as fts_span:
                fts_hits = await hybrid.fts_search(
                    req.document_ids, req.owner_id, req.query, dense_k, sub_queries=req.sub_queries
                )
                fts_span.update(hits=len(fts_hits))
            stages["fts_hits"] = len(fts_hits)
            fts_weight = settings.hybrid_fts_weight
            if req.sub_queries:
                fts_weight = max(fts_weight, 3.5)
            elif any(q in req.query for q in ('"', "'", "“", "”", "‘", "’")):
                fts_weight = max(fts_weight, 2.5)
            fusion_inputs.append((fts_hits, fts_weight))
        fusion_inputs.extend(extra_lanes)
        if graph_hits:
            fusion_inputs.append(
                (graph_hits, settings.retrieval_graph_lane_weight)
            )

        with tracing.span(trace_span, "rrf_fusion", lanes=len(fusion_inputs)) as fuse_span:
            fused = hybrid.reciprocal_rank_fusion(fusion_inputs, top_k=dense_k)
            fuse_span.update(fused=len(fused))
        stages["fused_hits"] = len(fused)
        candidates = fused
    else:
        candidates = dense_hits

    # --- Optional reranker ---
    rerank_target_k = (
        max(req.top_k, req.top_k * 2) if mmr_on else req.top_k
    )  # keep a pool for MMR to diversify from
    if rerank_on and candidates:
        with tracing.span(
            trace_span,
            "rerank",
            provider=settings.reranker_provider,
            model=settings.reranker_model,
            candidates=len(candidates),
        ) as rr_span:
            before_ids = [c.chunk_id for c in candidates[:rerank_target_k]]
            candidates = await reranker.rerank(req.query, candidates, top_k=rerank_target_k)
            after_ids = [c.chunk_id for c in candidates]
            moved = sum(
                1
                for i, cid in enumerate(after_ids)
                if i >= len(before_ids) or before_ids[i] != cid
            )
            rr_span.update(reordered=moved, final=len(candidates))
        stages["rerank_reordered"] = moved

    # --- Optional MMR diversification ---
    if mmr_on and candidates:
        with tracing.span(
            trace_span,
            "mmr",
            lambda_=settings.retrieval_mmr_lambda,
            candidates=len(candidates),
        ) as mmr_span:
            before_ids = [c.chunk_id for c in candidates[: req.top_k]]
            candidates = mmr_mod.mmr(
                candidates,
                top_k=req.top_k,
                lambda_=settings.retrieval_mmr_lambda,
            )
            after_ids = [c.chunk_id for c in candidates]
            mmr_moved = sum(
                1
                for i, cid in enumerate(after_ids)
                if i >= len(before_ids) or before_ids[i] != cid
            )
            mmr_span.update(reordered=mmr_moved, hits=len(candidates))
        stages["mmr_reordered"] = mmr_moved
    else:
        keep_k = max(req.top_k, 25 if req.sub_queries else req.top_k)
        candidates = candidates[:keep_k]

    # --- Adjacent Window / Section Expansion (Small-to-Big Retrieval) ---
    if candidates:
        force_expand = bool(req.sub_queries)
        expanded_chunks, exp_stats = await expand_chunk_windows(
            candidates,
            owner_id=req.owner_id,
            max_total_chunks=max(req.top_k, 28 if force_expand else req.top_k + 4),
            force_expand=force_expand,
        )
        candidates = expanded_chunks
        stages["window_expansion"] = exp_stats

    stages["final_hits"] = len(candidates)
    return RetrievalResult(chunks=candidates, stages=stages)
