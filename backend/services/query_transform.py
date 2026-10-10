"""Query transformations for improved recall.

Supported transforms (all flag-gated, all default OFF):

- **multi_query**  generate N paraphrases with distinct keywords so the
  dense retriever sees the same intent through multiple lexical lenses.
- **hyde**         ask the LLM to hallucinate a short ideal answer to the
  question; embed the hallucination rather than the question so the
  nearest-neighbor search lands in answer-shaped regions.
- **step_back**    produce a more general "step-back" question that
  retrieves broader context, useful for multi-hop reasoning.

Each transform returns a list of alternative query strings. Embedding
them is the caller's responsibility (this module stays provider-agnostic
and test-friendly).

All transforms degrade gracefully: a provider failure or missing API key
returns an empty list rather than raising, so retrieval still proceeds
on the original query.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from backend.services.llm import create_openai_text, gemini_text
from backend.settings import settings

logger = logging.getLogger("ragapp.query_transform")

_LINE_SPLIT_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*", re.MULTILINE)


@dataclass(slots=True)
class TransformedQuery:
    """One alternative query string produced by a named transform (e.g. HyDE, multi-query)."""

    text: str
    kind: str  # "multi_query" | "hyde" | "step_back"


def parse_transforms(raw: str) -> list[str]:
    """Parse a comma-separated `RETRIEVAL_QUERY_TRANSFORMS` setting into normalized kind names."""
    return [t.strip().lower() for t in (raw or "").split(",") if t.strip()]


def _parse_list(text: str, max_items: int) -> list[str]:
    """Parse a numbered / bulleted LLM response into deduped query strings."""
    if not text:
        return []
    # Try JSON first (robust when the model obeys "reply with a JSON array").
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            data = json.loads(stripped)
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()][:max_items]
        except json.JSONDecodeError:
            pass

    # Otherwise split on newlines + strip bullet/numeric prefixes.
    lines = [_LINE_SPLIT_RE.sub("", line).strip() for line in stripped.splitlines()]
    lines = [line for line in lines if line and len(line) > 3]
    # Drop duplicates while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= max_items:
            break
    return out


async def _call_llm(provider: str, api_key: str, model: str, prompt: str) -> str:
    """Minimal LLM call used for transforms. Returns "" on any failure."""
    if not api_key or not model:
        return ""
    messages = [
        {"role": "system", "content": "You are a concise query rewriter."},
        {"role": "user", "content": prompt},
    ]
    try:
        if provider == "openai":
            return await create_openai_text(api_key, model, messages)
        if provider == "deepseek":
            return await create_openai_text(api_key, model, messages, base_url="https://api.deepseek.com")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, gemini_text, api_key, model, messages)
    except Exception as exc:  # noqa: BLE001
        logger.warning("query_transform_llm_failed kind=%s err=%s", provider, exc)
        return ""


# ---- Individual transforms ---------------------------------------------

async def multi_query(
    question: str, *, provider: str, api_key: str, model: str, count: int
) -> list[TransformedQuery]:
    """Generate paraphrased questions to widen dense retrieval recall."""
    if count <= 0:
        return []
    prompt = (
        f"Generate {count} alternative phrasings of the following question. "
        "Each alternative must cover the same intent but use different keywords "
        "and synonyms. Reply with a numbered list only; no preamble.\n\n"
        f"Question: {question}"
    )
    text = await _call_llm(provider, api_key, model, prompt)
    return [TransformedQuery(text=q, kind="multi_query") for q in _parse_list(text, count)]


async def hyde(
    question: str, *, provider: str, api_key: str, model: str
) -> list[TransformedQuery]:
    """HyDE: hallucinate a short ideal answer to use as the retrieval query."""
    prompt = (
        "Write a concise, factual paragraph (3-5 sentences) that would be the "
        "ideal answer to the question below. Do not include caveats or meta "
        "commentary; only the answer body.\n\n"
        f"Question: {question}"
    )
    text = await _call_llm(provider, api_key, model, prompt)
    text = text.strip()
    if not text:
        return []
    return [TransformedQuery(text=text, kind="hyde")]


async def step_back(
    question: str, *, provider: str, api_key: str, model: str
) -> list[TransformedQuery]:
    """Generate a more general 'step-back' question."""
    prompt = (
        "Rewrite the following question as a single more general question that "
        "retrieves background context useful for answering the original. "
        "Reply with the new question only; no preamble.\n\n"
        f"Question: {question}"
    )
    text = await _call_llm(provider, api_key, model, prompt)
    text = text.strip().splitlines()[0] if text else ""
    if not text:
        return []
COMPARISON_MARKERS = [
    "so sánh", "khác nhau", "giống nhau", "đối chiếu", "phân biệt",
    "qua các thời kỳ", "qua các đại hội", "và cho biết", "đồng thời",
    "compare", "difference between", "similarities", "versus", " vs ",
]


def is_comparison_or_multihop(question: str, mode: str | None = None) -> bool:
    """Return True if question involves comparison, multi-entity analysis, or multi-hop logic."""
    if mode == "compare":
        return True
    lower = question.lower()
    return any(m in lower for m in COMPARISON_MARKERS)


async def decompose_query(
    question: str,
    *,
    provider: str,
    api_key: str,
    model: str,
) -> list[TransformedQuery]:
    """Decompose complex/comparison query into 2-4 distinct, self-contained sub-queries."""
    prompt = (
        "You are an expert retrieval query planner. The user asked a complex or comparison question:\n"
        f"\"{question}\"\n\n"
        "Break this question down into 2 to 4 distinct, standalone sub-queries so that a search engine can "
        "retrieve targeted evidence for every specific entity, document, time period, or perspective.\n"
        "Rules:\n"
        "1. Each sub-query must be fully self-contained (include exact names/dates, don't use pronouns like 'it', 'they', 'này').\n"
        "2. If comparing A and B, ensure at least one sub-query targets A, one targets B, and one targets the transition/relation.\n"
        "3. Reply with ONLY a valid JSON array of strings, e.g. [\"query 1\", \"query 2\", \"query 3\"]. No markdown ticks or other text."
    )
    raw = await _call_llm(provider, api_key, model, prompt)
    queries = _parse_list(raw, max_items=4)
    if not queries:
        # Fallback to connector / clause splitting
        split_patterns = r"[?.;!]\s*|\s+(?:và cho biết|đồng thời|và nêu|cho biết|và xác định|so với|so sánh với)\s+"
        parts = [c.strip() for c in re.split(split_patterns, question) if len(c.strip()) > 8]
        if len(parts) > 1:
            queries = parts[:4]

    return [TransformedQuery(text=q, kind="decomposition") for q in queries]


# ---- Orchestrator ------------------------------------------------------

async def transform(
    question: str,
    *,
    provider: str,
    api_key: str,
    model: str,
    mode: str | None = None,
) -> list[TransformedQuery]:
    """Run all enabled transforms in parallel. Automatically decomposes comparison/multi-hop queries."""
    tasks: list[asyncio.Task[list[TransformedQuery]]] = []

    # 1. Automatic Decomposition for comparison, multi-entity, or multi-hop questions
    if is_comparison_or_multihop(question, mode):
        tasks.append(
            asyncio.create_task(
                decompose_query(
                    question,
                    provider=provider,
                    api_key=api_key,
                    model=model,
                )
            )
        )

    # 2. General query transformations if flag is enabled
    if settings.retrieval_query_transforms_enabled:
        kinds = parse_transforms(settings.retrieval_query_transforms)
        if "multi_query" in kinds:
            tasks.append(
                asyncio.create_task(
                    multi_query(
                        question,
                        provider=provider,
                        api_key=api_key,
                        model=model,
                        count=settings.retrieval_multi_query_count,
                    )
                )
            )
        if "hyde" in kinds:
            tasks.append(
                asyncio.create_task(
                    hyde(question, provider=provider, api_key=api_key, model=model)
                )
            )
        if "step_back" in kinds:
            tasks.append(
                asyncio.create_task(
                    step_back(question, provider=provider, api_key=api_key, model=model)
                )
            )
    if not tasks:
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: list[TransformedQuery] = []
    seen: set[str] = {question.strip().lower()}
    for result in results:
        if isinstance(result, Exception):
            logger.warning("query_transform_failed err=%s", result)
            continue
        for tq in result:
            key = tq.text.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(tq)
    return out
