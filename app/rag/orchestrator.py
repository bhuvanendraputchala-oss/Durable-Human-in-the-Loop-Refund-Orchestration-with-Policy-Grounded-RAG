from __future__ import annotations

import logging
import re
from typing import List

from langfuse.decorators import observe, langfuse_context
from langchain_openai import ChatOpenAI

from .retriever import retrieve_similarity, retrieve_mmr

logger = logging.getLogger(__name__)

# Handles: "0 9", "0: 9", "0: 9.5", "[0] 9", "0 - 9", "0. 9"
_SCORE_RE = re.compile(r"^\s*\[?(\d+)\]?[\s:.\-]+(\d+(?:\.\d+)?)")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_MMR_ISSUE_TYPES = {"defective_product", "wrong_item", "missing_item"}


@observe()
def _plan_queries(ticket_text: str, issue_type: str) -> List[str]:
    prompt = (
        "You are a policy retrieval planner.\n"
        f"Issue type: {issue_type}\n"
        f"Customer message: {ticket_text}\n\n"
        "Generate 3 short, distinct search queries to retrieve the most relevant "
        "policy sections. Return each query on its own line with no numbering or bullets."
    )
    try:
        result = llm.invoke(prompt)
        lines = [l.strip() for l in result.content.strip().splitlines() if l.strip()]
    except Exception as exc:
        logger.warning("_plan_queries: LLM call failed (%s); falling back to original query.", exc)
        lines = []
    queries = list(dict.fromkeys([ticket_text] + lines))
    langfuse_context.update_current_observation(
        input={"ticket_text": ticket_text, "issue_type": issue_type},
        output={"planned_queries": queries},
    )
    return queries


def _select_retriever(issue_type: str):
    if issue_type in _MMR_ISSUE_TYPES:
        return retrieve_mmr
    return retrieve_similarity


@observe()
def _rescore(ticket_text: str, docs: list) -> list:
    if not docs:
        return []

    excerpts = "\n\n".join(f"[{i}] {d.page_content[:300]}" for i, d in enumerate(docs))
    prompt = (
        f"Rate each policy excerpt for relevance to this customer issue (0-10).\n"
        f"Issue: {ticket_text}\n\n"
        f"Excerpts:\n{excerpts}\n\n"
        "Reply with one line per excerpt in the format: INDEX SCORE\n"
        "Example:\n0 9\n1 4\n2 7"
    )

    scores: dict[int, float] = {}
    try:
        result = llm.invoke(prompt)
        for line in result.content.strip().splitlines():
            m = _SCORE_RE.match(line)
            if m:
                idx, score = int(m.group(1)), float(m.group(2))
                if 0 <= idx < len(docs):
                    scores[idx] = score

        if not scores:
            logger.warning(
                "_rescore: no parseable scores in LLM response; preserving original order. Response was: %r",
                result.content[:200],
            )
    except Exception as exc:
        logger.warning("_rescore: LLM call failed (%s); preserving original order.", exc)

    ranked = [
        doc
        for _, doc in sorted(enumerate(docs), key=lambda x: scores.get(x[0], 0.0), reverse=True)
    ]
    langfuse_context.update_current_observation(
        metadata={
            "scores": scores,
            "input_doc_count": len(docs),
            "output_doc_count": len(ranked),
            "fallback_order": not scores,
        }
    )
    return ranked


@observe()
def kb_orchestrator(state: dict) -> dict:
    ticket_text: str = state.get("ticket_text", "")
    issue_type: str = state.get("issue_type", "other")

    queries = _plan_queries(ticket_text, issue_type)

    retriever_fn = _select_retriever(issue_type)
    retriever_type = retriever_fn.__name__

    seen: set = set()
    all_docs: list = []
    for q in queries:
        for doc in retriever_fn(q, k=4):
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    ranked_docs = _rescore(ticket_text, all_docs)[:6]

    citations = [
        {
            "source": d.metadata.get("source", "unknown"),
            "chunk_id": d.metadata.get("chunk_id", "unknown"),
            "start_char": d.metadata.get("start_char", 0),
            "snippet": d.page_content[:200],
        }
        for d in ranked_docs
    ]

    langfuse_context.update_current_observation(
        input={"ticket_text": ticket_text, "issue_type": issue_type},
        output={"citations": citations},
        metadata={
            "retriever_type": retriever_type,
            "planned_queries": queries,
            "doc_ids": [c["chunk_id"] for c in citations],
            "citation_spans": [
                {"chunk_id": c["chunk_id"], "source": c["source"], "start_char": c["start_char"]}
                for c in citations
            ],
            "total_candidates": len(all_docs),
            "returned_count": len(ranked_docs),
        },
    )

    return {
        "rag_docs": [d.page_content for d in ranked_docs],
        "citations": citations,
    }


def run_policy_rag(query: str) -> dict:
    from .retriever import retrieve
    docs = retrieve(query)
    return {
        "documents": [d.page_content for d in docs],
        "citations": [
            {"source": d.metadata.get("source", "unknown"), "snippet": d.page_content[:200]}
            for d in docs
        ],
    }
