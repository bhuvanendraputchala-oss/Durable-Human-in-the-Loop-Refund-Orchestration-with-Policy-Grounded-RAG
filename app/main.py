from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import json, os, re
from langfuse.decorators import observe, langfuse_context
from typing import Any, Dict, List, Optional, TypedDict
from langchain_core.messages import AnyMessage
from dotenv import load_dotenv
import time
load_dotenv()

app = FastAPI(title="Phase 1 Mock API")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOCK_DIR = os.path.join(ROOT, "mock_data")

def load(name):
    with open(os.path.join(MOCK_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)

ORDERS = load("orders.json")
ISSUES = load("issues.json")
REPLIES = load("replies.json")

from app.graph import build_graph
GRAPH = build_graph()


class TriageInput(BaseModel):
    messages: List[AnyMessage]
    ticket_text: str
    order_id: str | None = None
    messages: list[dict] = []
    issue_type: str | None = None
    evidence: dict = {}
    recommendation: str | None = None
    needs_admin: bool | None = None
    admin_decision: str | None = None
    admin_notes: str | None = None
    reply_draft: str | None = None

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/orders/get")
def orders_get(order_id: str = Query(...)):
    for o in ORDERS:
        if o["order_id"] == order_id: return o
    raise HTTPException(status_code=404, detail="Order not found")

@app.get("/orders/search")
def orders_search(customer_email: str | None = None, q: str | None = None):
    matches = []
    for o in ORDERS:
        if customer_email and o["email"].lower() == customer_email.lower():
            matches.append(o)
        elif q and (o["order_id"].lower() in q.lower() or o["customer_name"].lower() in q.lower()):
            matches.append(o)
    return {"results": matches}

@app.post("/classify/issue")
def classify_issue(payload: dict):
    text = payload.get("ticket_text", "").lower()
    for rule in ISSUES:
        if rule["keyword"] in text:
            return {"issue_type": rule["issue_type"], "confidence": 0.85}
    return {"issue_type": "unknown", "confidence": 0.1}

def render_reply(issue_type: str, order):
    template = next((r["template"] for r in REPLIES if r["issue_type"] == issue_type), None)
    if not template: template = "Hi {{customer_name}}, we are reviewing order {{order_id}}."
    return template.replace("{{customer_name}}", order.get("customer_name","Customer")).replace("{{order_id}}", order.get("order_id",""))

@app.post("/reply/draft")
def reply_draft(payload: dict):
    return {"reply_text": render_reply(payload.get("issue_type"), payload.get("order", {}))}


@app.post("/triage/invoke")
@observe()
def triage_invoke(body: TriageInput):
    state = body.model_dump()

    langfuse_context.update_current_trace(
        name="triage_invoke",
        input=state,
        metadata={
            "order_id": state.get("order_id"),
            "issue_type": state.get("issue_type"),
            "needs_admin": state.get("needs_admin"),
            "admin_decision": state.get("admin_decision"),
        },
        tags=["phase1", "triage"],
    )

    result = GRAPH.invoke(state)

    langfuse_context.update_current_trace(output=result)
    return result
