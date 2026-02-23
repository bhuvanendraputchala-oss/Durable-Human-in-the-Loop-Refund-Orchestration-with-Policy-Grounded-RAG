from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from langfuse.decorators import observe, langfuse_context
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from app.rag.orchestrator import kb_orchestrator
from app.payments import refund_preview as payments_preview, refund_commit as payments_commit
from .state import TriageState
from .templates import render_reply
from .tools import fetch_order

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_saver_ctx = None
_saver = None

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOCK_DIR = os.path.join(ROOT, "mock_data")
ORDER_ID_REGEX = re.compile(r"\b(ORD\d{4})\b", re.IGNORECASE)

fetch_order_node = ToolNode([fetch_order])


def load_json(filename: str) -> Any:
    path = os.path.join(MOCK_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filename}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error reading {filename}: {e}") from e


issue_keywords = load_json("issues.json")


def append_issue_keywords(state: TriageState, role: str, text: str) -> None:
    msgs = state.get("messages") or []
    if role == "customer":
        msgs.append(HumanMessage(content=text))
    else:
        msgs.append(AIMessage(content=text))
    state["messages"] = msgs


@observe()
def ingest(state: TriageState) -> TriageState:
    ticket = (state.get("ticket_text") or "").strip()
    if not ticket:
        append_issue_keywords(state, "assistant", "I did not receive a ticket. Please paste the customer message.")
        return state

    state["ticket_text"] = ticket
    state["evidence"] = state.get("evidence") or {}

    msgs = state.get("messages") or []
    if not msgs:
        append_issue_keywords(state, "customer", ticket)
    else:
        customer_message_exists = any(
            isinstance(msg, HumanMessage) and msg.content == ticket
            for msg in msgs
        )
        if not customer_message_exists:
            append_issue_keywords(state, "customer", ticket)

    if not state.get("order_id"):
        m = ORDER_ID_REGEX.search(ticket)
        if m:
            state["order_id"] = m.group(1).upper()

    return state


@observe()
def classify_issue(state: TriageState) -> TriageState:
    if state.get("issue_type"):
        return state

    text = (state.get("ticket_text") or "").lower()
    issue_type: Optional[str] = None

    for row in issue_keywords:
        kw = (row.get("keyword") or "").lower()
        if kw and kw in text:
            issue_type = row.get("issue_type")
            break

    if not issue_type:
        issue_type = "refund_request" if "refund" in text else "defective_product"

    state["issue_type"] = issue_type
    append_issue_keywords(state, "assistant", f"The issue is classified as {issue_type}.")
    return state


@observe()
def request_fetch_order(state: TriageState) -> TriageState:
    state.setdefault("evidence", {})
    order_id = state.get("order_id")

    if not order_id:
        state["evidence"]["order"] = {"found": False, "reason": "No order ID provided"}
        append_issue_keywords(state, "assistant", "Order id is missing. Please provide the order ID.")
        return state

    msgs = state.get("messages") or []
    msgs.append(
        AIMessage(
            content="Fetching order details.",
            tool_calls=[
                {
                    "name": "fetch_order",
                    "args": {"order_id": order_id},
                    "id": "call_fetch_order_1",
                    "type": "tool_call",
                }
            ],
        )
    )
    state["messages"] = msgs
    return state


@observe()
def store_order_evidence(state: TriageState) -> TriageState:
    state.setdefault("evidence", {})
    msgs = state.get("messages") or []

    for msg in reversed(msgs):
        if isinstance(msg, ToolMessage) and msg.name == "fetch_order":
            content = msg.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except Exception:
                    pass
            state["evidence"]["order"] = content
            return state

    return state


@observe()
def propose_recommendation(state: TriageState) -> TriageState:
    if state.get("recommendation"):
        return state

    issue_type = state.get("issue_type") or "other"
    order_info = state.get("evidence", {}).get("order", {}) or {}

    if not isinstance(order_info, dict) or not order_info.get("found"):
        rec = "Ask the customer for the correct order id and confirm their email address."
    else:
        if issue_type == "refund_request":
            rec = "Confirm eligibility and initiate refund. Share expected timeline."
        elif issue_type == "late_delivery":
            rec = "Share current shipping status and set expectation for delivery timing."
        elif issue_type == "missing_item":
            rec = "Open a missing item investigation and offer replacement or reship."
        elif issue_type == "damaged_item":
            rec = "Apologize and offer replacement. Ask for photo if needed."
        elif issue_type == "duplicate_charge":
            rec = "Confirm duplicate charge and refund the extra amount."
        elif issue_type == "wrong_item":
            rec = "Arrange replacement and provide return instructions for the incorrect item."
        elif issue_type == "defective_product":
            rec = "Confirm warranty coverage and offer replacement or repair."
        else:
            rec = "Escalate to a human agent for further review."

    state["recommendation"] = rec
    state["needs_admin"] = True
    append_issue_keywords(state, "assistant", f"Proposed action: {rec}")
    append_issue_keywords(state, "assistant", f"Needs admin: {state['needs_admin']}")
    return state


@observe()
def route_after_propose(state: TriageState) -> str:
    if state.get("refund_approved"):
        return "commit_remedy"
    return "draft_reply"


@observe()
def draft_reply(state: TriageState) -> TriageState:
    if state.get("reply_draft"):
        return state

    decision = (state.get("admin_decision") or "").strip().lower()
    issue_type = state.get("issue_type") or "other"
    order_payload = (state.get("evidence") or {}).get("order") or {}

    if decision == "reject":
        reply = (
            "Thanks for reaching out. I reviewed your request, but I need a bit more information before I can proceed. "
            "Can you confirm what went wrong and share any details like photos, error messages, or what troubleshooting you tried? "
            "If needed, I will escalate this to a specialist."
        )
        state["reply_draft"] = reply
        append_issue_keywords(state, "assistant", reply)
        return state

    if isinstance(order_payload, dict) and order_payload.get("found"):
        order = order_payload["order"]
        reply = render_reply(issue_type, order)
    else:
        reply = "Hi there, can you share your order id so I can look this up and help you quickly?"

    citations_block = _format_citations(state.get("citations") or [])
    reply = reply + citations_block

    state["reply_draft"] = reply
    append_issue_keywords(state, "assistant", reply)
    return state


@observe()
def policy_evaluator(state: TriageState) -> TriageState:
    issue_text = state.get("ticket_text", "")
    recommendation = state.get("recommendation", "")

    rag_docs = state.get("rag_docs") or []
    context = "\n\n".join(rag_docs) if rag_docs else "No policy excerpts retrieved."

    prompt = f"""You are a policy compliance evaluator.

Customer issue:
{issue_text}

Proposed action:
{recommendation}

Relevant policy excerpts:
{context}

Determine:
- Is the proposed action allowed under the cited policies?
- If yes, quote the specific policy language that permits it.
- If no, quote the specific policy language that prohibits it.

You must reference policy language from the excerpts above.

On the very last line of your response, write exactly one of:
VERDICT: ALLOWED
VERDICT: DENIED"""

    try:
        decision = llm.invoke(prompt)
        content = decision.content
    except Exception as exc:
        logger.error("policy_evaluator: LLM call failed: %s", exc)
        return {
            "policy_decision": f"Policy evaluation unavailable ({exc}). Defaulting to ALLOWED.",
            "policy_allowed": True,
        }

    policy_allowed: bool = True
    verdict_found: bool = False
    for raw_line in reversed(content.splitlines()):
        stripped = raw_line.strip()
        if stripped.upper().startswith("VERDICT:"):
            verdict_value = stripped[len("VERDICT:"):].strip().upper()
            policy_allowed = verdict_value == "ALLOWED"
            verdict_found = True
            break

    if not verdict_found:
        logger.warning(
            "policy_evaluator: no VERDICT line found; defaulting to ALLOWED. Response tail: %r",
            content[-300:],
        )

    citations = state.get("citations") or []
    langfuse_context.update_current_observation(
        input={"ticket_text": issue_text, "recommendation": recommendation},
        output={"policy_decision": content, "policy_allowed": policy_allowed},
        metadata={
            "policy_allowed": policy_allowed,
            "doc_ids": [c.get("chunk_id", "unknown") for c in citations],
            "citation_spans": [
                {
                    "chunk_id": c.get("chunk_id", "unknown"),
                    "source": c.get("source", "unknown"),
                    "start_char": c.get("start_char", 0),
                }
                for c in citations
            ],
            "citation_count": len(citations),
        },
    )

    return {"policy_decision": content, "policy_allowed": policy_allowed}


_REFUND_ISSUE_TYPES = {"refund_request", "duplicate_charge", "late_delivery", "missing_item"}


def _format_citations(citations: list, max_citations: int = 3) -> str:
    if not citations:
        return ""
    lines = []
    for c in citations[:max_citations]:
        source = c.get("source", "unknown")
        snippet = (c.get("snippet") or "").strip()[:120]
        lines.append(f"  • {source}: \"{snippet}\"")
    return "\n\nPolicy citations:\n" + "\n".join(lines)


@observe()
def propose_remedy(state: TriageState) -> TriageState:
    policy_decision = state.get("policy_decision") or ""

    if not state.get("policy_allowed", True):
        recommendation = "Refund not permitted under refund policy. Offer warranty evaluation."
        append_issue_keywords(state, "assistant", recommendation)
        return {
            "policy_decision": policy_decision,
            "recommendation": recommendation,
            "needs_admin": False,
        }

    order_data = (state.get("evidence") or {}).get("order") or {}
    order = order_data.get("order", {}) if isinstance(order_data, dict) else {}
    order_id = (
        state.get("order_id")
        or (order.get("order_id") if isinstance(order, dict) else None)
        or "unknown"
    )
    amount = float(order.get("total_amount", 0.0)) if isinstance(order, dict) else 0.0

    preview = payments_preview(order_id, amount)

    citations_block = _format_citations(state.get("citations") or [])
    append_issue_keywords(
        state,
        "assistant",
        f"Refund preview ready: ${preview['amount']:.2f} via {preview['method']} "
        f"(ETA {preview['eta']}). Awaiting human approval.{citations_block}",
    )
    return {
        "refund_preview": preview,
        "policy_decision": policy_decision,
        "recommendation": "Refund permitted. Proceed with preview.",
        "needs_admin": True,
    }


@observe()
def commit_remedy(state: TriageState) -> TriageState:
    preview = state.get("refund_preview") or {}
    order_id = preview.get("order_id", "unknown")
    preview_id = preview.get("preview_id", "")

    if state.get("refund_approved"):
        result = payments_commit(order_id, preview_id)
        append_issue_keywords(state, "assistant", f"Refund committed. Transaction ID: {result['transaction_id']}")
    else:
        reason = state.get("refund_reject_reason") or "Refund rejected by human reviewer."
        result = {"status": "rejected", "reason": reason}
        append_issue_keywords(state, "assistant", f"Refund rejected. Reason: {reason}")

    return {"remedy_result": result, "needs_admin": False}


def route_after_admin(state: TriageState) -> str:
    decision = (state.get("admin_decision") or "").strip().lower()
    refund_eligible = state.get("issue_type") in _REFUND_ISSUE_TYPES

    if refund_eligible and not decision:
        return "propose_remedy"
    if decision == "approve":
        return "commit_remedy"
    return "draft_reply"


def route_after_ingest(state: TriageState) -> str:
    ticket = (state.get("ticket_text") or "").strip()
    return END if not ticket else "classify_issue"


def route_after_classify(state: TriageState) -> str:
    return "request_fetch_order" if state.get("order_id") else "propose_recommendation"


@observe()
def admin_review(state: TriageState) -> TriageState:
    decision = (state.get("admin_decision") or "").strip().lower()
    if decision not in ["approve", "reject"]:
        state["needs_admin"] = True
        return state
    state["needs_admin"] = False
    return state


def build_graph():
    sg = StateGraph(TriageState)

    sg.add_node("ingest", ingest)
    sg.add_node("classify_issue", classify_issue)
    sg.add_node("request_fetch_order", request_fetch_order)
    sg.add_node("fetch_order", fetch_order_node)
    sg.add_node("store_order_evidence", store_order_evidence)
    sg.add_node("propose_recommendation", propose_recommendation)
    sg.add_node("kb_orchestrator", kb_orchestrator)
    sg.add_node("policy_evaluator", policy_evaluator)
    sg.add_node("admin_review", admin_review)
    sg.add_node("propose_remedy", propose_remedy)
    sg.add_node("commit_remedy", commit_remedy)
    sg.add_node("draft_reply", draft_reply)

    sg.add_edge(START, "ingest")
    sg.add_conditional_edges("ingest", route_after_ingest, {"classify_issue": "classify_issue", END: END})
    sg.add_conditional_edges("classify_issue", route_after_classify, {"request_fetch_order": "request_fetch_order", "propose_recommendation": "propose_recommendation"})
    sg.add_edge("request_fetch_order", "fetch_order")
    sg.add_edge("fetch_order", "store_order_evidence")
    sg.add_edge("store_order_evidence", "propose_recommendation")
    sg.add_edge("propose_recommendation", "kb_orchestrator")
    sg.add_edge("kb_orchestrator", "policy_evaluator")
    sg.add_edge("policy_evaluator", "admin_review")
    sg.add_conditional_edges("admin_review", route_after_admin, {"propose_remedy": "propose_remedy", "commit_remedy": "commit_remedy", "draft_reply": "draft_reply"})
    sg.add_conditional_edges("propose_remedy", route_after_propose, {"commit_remedy": "commit_remedy", "draft_reply": "draft_reply"})
    sg.add_edge("commit_remedy", "draft_reply")
    sg.add_edge("draft_reply", END)

    global _saver_ctx, _saver
    if _saver is None:
        uri = os.getenv("POSTGRES_URI")
        if not uri:
            raise RuntimeError(
                "POSTGRES_URI is not set. "
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
            )
        _saver_ctx = PostgresSaver.from_conn_string(uri)
        _saver = _saver_ctx.__enter__()
        _saver.setup()

    return sg.compile(
        checkpointer=_saver,
        interrupt_after=["propose_remedy"],
    )
