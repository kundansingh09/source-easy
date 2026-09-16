"""LLM-based reranking of the top hybrid-search candidates.

Design: hybrid search (dense + sparse RRF/DBSF) already narrows ~1300
companies down to a shortlist. This module only reorders that shortlist -
it never sees the full catalog - so cost stays flat regardless of how large
the supplier database grows.
"""

import json
import os

from openai import OpenAI

RERANK_MODEL = os.environ.get("RERANK_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a semiconductor supply-chain sourcing analyst. \
A procurement buyer has issued a search query against a database of SEMICON \
trade-show exhibitors. You will be given the query and a shortlist of \
candidate companies (name, HQ, category tags, and their own description \
where available).

Score each candidate 0-100 on how well it actually matches what the buyer \
is asking for, using these rules:

1. Ground every score in the text given. Do not assume a company offers a \
capability just because its name or category sounds relevant - if the \
description doesn't support it, score it lower and say why in one phrase.
2. A precise category match with no description is worth more than a vague \
or off-topic description - but less than a description that explicitly \
confirms the capability.
3. Prefer specific technical fit (equipment type, process step, materials, \
tolerances, purity grade) over generic "semiconductor equipment supplier" \
matches.
4. Do not let company size, fame, or HQ location influence the score unless \
the query explicitly asks about them.
5. If NONE of the candidates are a good match, say so honestly with low \
scores across the board rather than forcing a confident-looking ranking.

Return ONLY a JSON array, no prose, no markdown fences:
[{"id": <candidate id>, "score": <0-100 integer>, "reason": "<max 12 words>"}, ...]
Include every candidate id exactly once."""


def _format_candidates(candidates):
    lines = []
    for c in candidates:
        cats = ", ".join((c.get("categories_l2") or c.get("categories_l1") or [])[:5])
        about = c.get("about") or ""
        if about == "Semiconductor technology and equipment supplier.":
            about = "(no description provided)"
        lines.append(
            f"id={c['id']} | {c['company_name']} | HQ: {c.get('hq_country', 'Unknown')} "
            f"| Categories: {cats or 'none listed'} | About: {about}"
        )
    return "\n".join(lines)


def llm_rerank(query, candidates, top_n=None, client=None):
    """candidates: list of dicts from SourcingSearchEngine.search(), each
    must additionally carry a stable 'id' key. Returns the same dicts
    reordered, with 'llm_score' and 'llm_reason' added. On any failure
    (no API key, network error, malformed response) falls back to the
    original hybrid-search order unchanged, so a rerank outage never
    breaks the search itself."""
    if not candidates:
        return candidates

    client = client or OpenAI()  # reads OPENAI_API_KEY from env

    try:
        resp = client.chat.completions.create(
            model=RERANK_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nCandidates:\n{_format_candidates(candidates)}"},
            ],
            # A bare JSON array isn't valid for response_format="json_object"
            # (that mode requires a top-level object), so the array format is
            # enforced via the prompt instead and parsed defensively below.
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        scored = {row["id"]: row for row in json.loads(raw)}
    except Exception as e:
        print(f"[llm_rerank] falling back to hybrid order - {e}")
        return candidates[:top_n] if top_n else candidates

    ranked = sorted(
        candidates,
        key=lambda c: scored.get(c["id"], {}).get("score", -1),
        reverse=True,
    )
    for c in ranked:
        hit = scored.get(c["id"])
        c["llm_score"] = hit["score"] if hit else None
        c["llm_reason"] = hit.get("reason") if hit else None

    return ranked[:top_n] if top_n else ranked