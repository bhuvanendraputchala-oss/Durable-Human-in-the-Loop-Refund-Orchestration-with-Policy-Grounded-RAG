from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional
from langfuse.decorators import observe, langfuse_context
from langgraph.graph import StateGraph, START, END

from .state import TriageState
from .templates import render_reply
from .tools import fetch_order

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOCK_DIR = os.path.join(ROOT, "mock_data")
ORDER_ID_REGEX = re.compile(r"\b(ORD\d{4})\b", re.IGNORECASE)

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

def append_issue_keywords(state:TriageState, role:str, message:str) -> None:
    msgs=state.get("messages") or []
    msgs.append({"role": role, "content": message})
    state["messages"] = msgs


@observe()
def ingest(state:TriageState) -> TriageState:
    ticket=(state.get("ticket_text") or "").strip()
    if not ticket:
        append_issue_keywords(state, "assistant", "I did not receive a ticket. Please paste the customer message.")
        return state
    
    state["ticket_text"] = ticket
    state["evidence"] = state.get("evidence") or {}

    # Only append customer message if it hasn't been ingested yet
    msgs = state.get("messages") or []
    customer_message_exists = any(
        msg.get("role") == "customer" and msg.get("content") == ticket 
        for msg in msgs
    )
    
    if not customer_message_exists:
        append_issue_keywords(state, "customer", ticket)

    if not state.get("order_id"):
        m=ORDER_ID_REGEX.search(ticket)
        if m:
            state["order_id"] = m.group(1).upper()
    return state

@observe()
def classify(state:TriageState) -> TriageState:
    # Skip if already classified
    if state.get("issue_type"):
        return state
    
    text=(state.get("ticket_text") or "").lower()
    issue_type:Optional[str]=None
    for row in issue_keywords:
        kw=row["keyword"].lower()
        if kw in text:
            issue_type=row["issue_type"]
            break
    if not issue_type:
        issue_type="refund_request" if "refund" in text else "defective_product"
    state["issue_type"]=issue_type
    append_issue_keywords(state,"assistant", f"The issue is classified as {issue_type}.")
    return state

@observe()
def fetch_order_info(state:TriageState) -> TriageState:
    # Skip if order info already fetched
    if state.get("evidence", {}).get("order"):
        return state
    
    order_id=state.get("order_id")
    if not order_id:
        state["evidence"]["order"]= {"found": False, 'reason': "No order ID provided"}
        append_issue_keywords(state,"assistant", "Order id is missing. Please provide the order ID.")
        return state
    result=fetch_order.invoke({"order_id": order_id})
    state["evidence"]["order"]=result
    append_issue_keywords(state,"assistant", f"Fetched order details for {order_id}.")
    return state

@observe()
def propose_recommendation(state:TriageState) -> TriageState:
    # Skip if recommendation already exists
    if state.get("recommendation"):
        return state
    
    issue_type=state.get("issue_type") or "other"
    order_info=state.get("evidence", {}).get("order", {})

    if not order_info.get("found"):
        rec="Ask the customer for the correct order id and confirm their email address."
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
    state["recommendation"]=rec
    state["needs_admin"]=True
    append_issue_keywords(state,"assistant",f"Proposed action: {rec}")
    append_issue_keywords(state,"assistant",f"Needs admin: {state['needs_admin']}")
    return state


@observe()
def admin_review(state:TriageState) -> TriageState:
    decision=(state.get("admin_decision") or "").strip().lower()
    if decision not in ["approve", "reject"]:
        state["needs_admin"]=True
        return state
    
    state["needs_admin"]=False
    notes=(state.get("admin_notes") or "").strip()
    append_issue_keywords(state,"admin",f"Decision: {decision}. Notes: {notes}")

    if decision == "reject":
        state["recommendation"] = "Ask for clarification and escalate to a human agent."
    return state

@observe()
def draft_reply(state: TriageState) -> TriageState:
    # Skip if reply already drafted
    if state.get("reply_draft"):
        return state
    
    issue_type = state.get("issue_type") or "other"
    order_payload = (state.get("evidence") or {}).get("order") or {}

    if order_payload.get("found"):
        order = order_payload["order"]
        reply = render_reply(issue_type, order)
    else:
        reply = "Hi there, can you share your order id so I can look this up and help you quickly?"

    state["reply_draft"] = reply
    append_issue_keywords(state, "assistant", reply)
    return state


def route_after_classify(state: TriageState) -> str:
    if state.get("order_id"):
        return "fetch_order"
    return "propose_recommendation"


def route_after_admin(state: TriageState) -> str:
    if state.get("needs_admin"):
        return END
    return "draft_reply"


def build_graph():
    sg = StateGraph(TriageState)

    sg.add_node("ingest", ingest)
    sg.add_node("classify_issue", classify)
    sg.add_node("fetch_order", fetch_order_info)
    sg.add_node("propose_recommendation", propose_recommendation)
    sg.add_node("admin_review", admin_review)
    sg.add_node("draft_reply", draft_reply)

    sg.set_entry_point("ingest")
    sg.add_edge("ingest", "classify_issue")

    sg.add_conditional_edges(
        "classify_issue",
        route_after_classify,
        {"fetch_order": "fetch_order", "propose_recommendation": "propose_recommendation"},
    )

    sg.add_edge("fetch_order", "propose_recommendation")
    sg.add_edge("propose_recommendation", "admin_review")

    sg.add_conditional_edges(
        "admin_review",
        route_after_admin,
        {"draft_reply": "draft_reply", END: END},
    )

    sg.add_edge("draft_reply", END)

    return sg.compile()


