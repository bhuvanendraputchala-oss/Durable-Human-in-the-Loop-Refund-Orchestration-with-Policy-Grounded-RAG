from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import json, os, re, uuid
from langfuse.decorators import observe, langfuse_context
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import time

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = [v for v in ("OPENAI_API_KEY", "POSTGRES_URI") if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
        )
    yield


app = FastAPI(title="Phase 1 Mock API", lifespan=lifespan)
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
    thread_id: str | None = None


class ResumeInput(BaseModel):
    thread_id: str
    approved: bool
    reason: str | None = None


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
    thread_id = state.pop("thread_id", None) or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    langfuse_context.update_current_trace(
        name="triage_invoke",
        input=state,
        metadata={
            "thread_id": thread_id,
            "order_id": state.get("order_id"),
            "issue_type": state.get("issue_type"),
            "needs_admin": state.get("needs_admin"),
            "admin_decision": state.get("admin_decision"),
        },
        tags=["phase1", "triage"],
    )

    try:
        result = GRAPH.invoke(state, config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result.get("refund_preview") and not result.get("admin_decision"):
        return {
            "status": "awaiting_admin",
            "thread_id": thread_id,
            "state": result,
        }
    return {
        "status": "completed",
        "thread_id": thread_id,
        "state": result,
    }


@app.post("/triage/resume")
@observe()
def triage_resume(body: ResumeInput):

    config = {"configurable": {"thread_id": body.thread_id}}

    langfuse_context.update_current_trace(
        name="triage_resume",
        input={
            "thread_id": body.thread_id,
            "approved": body.approved,
            "reason": body.reason,
        },
        tags=["phase1", "triage", "resume"],
    )

    try:
        GRAPH.update_state(
            config,
            {
                "admin_decision": "approve" if body.approved else "reject",
                "refund_approved": body.approved,
                "refund_reject_reason": body.reason or "",
            },
        )
        result = GRAPH.invoke(None, config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    langfuse_context.update_current_trace(output=result)

    return {
        "status": "completed",
        "state": result,
    }

@app.post("/triage/admin")
def admin_resume(payload: dict):
    thread_id = payload["thread_id"]
    decision = payload["decision"]

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    GRAPH.update_state(
        config,
        {
            "admin_decision": decision
        },
    )
    result = GRAPH.invoke(None, config=config)

    return {
        "status": "completed",
        "state": result,
    }