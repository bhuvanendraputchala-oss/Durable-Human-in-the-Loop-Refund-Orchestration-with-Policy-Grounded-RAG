import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph import build_graph


def test_persist_interrupt_resume():
    """Start run, simulate process stop, restart, recover from Postgres, resume."""
    ticket = "I was charged twice for order ORD1003."
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    graph_a = build_graph()
    graph_a.invoke({"ticket_text": ticket, "messages": []}, config=config)

    snap = graph_a.get_state(config)
    assert snap.next, "Graph must be interrupted at propose_remedy"
    del graph_a

    graph_b = build_graph()
    snap_b = graph_b.get_state(config)
    assert snap_b.next, "State must persist in Postgres after simulated stop"
    assert snap_b.values.get("issue_type") == "duplicate_charge"
    assert snap_b.values.get("order_id") == "ORD1003"

    graph_b.update_state(config, {"admin_decision": "approve", "refund_approved": True})
    result = graph_b.invoke(None, config=config)

    assert result.get("remedy_result", {}).get("status") == "committed", (
        f"Expected remedy committed, got: {result.get('remedy_result')!r}"
    )


def test_human_approval_gate():
    """Graph pauses at propose_remedy; human approval triggers commit_remedy."""
    ticket = "Refund for my 4K Monitor (ORD1011)."
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    graph = build_graph()
    graph.invoke({"ticket_text": ticket, "messages": []}, config=config)

    snap = graph.get_state(config)
    assert snap.next, "Graph must pause for human approval"
    assert snap.values.get("refund_preview"), "refund_preview must be populated before approval"

    graph.update_state(config, {"admin_decision": "approve", "refund_approved": True})
    result = graph.invoke(None, config=config)

    remedy = result.get("remedy_result") or {}
    assert remedy.get("status") == "committed"
    assert remedy.get("transaction_id", "").startswith("txn_"), (
        f"transaction_id should start with 'txn_', got {remedy.get('transaction_id')!r}"
    )


def test_rag_citations_in_remedy_and_reply():
    """RAG citations appear in the citations list and in the final reply_draft."""
    ticket = "My watch from ORD1004 stopped working within 10 days."
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    graph = build_graph()
    result = graph.invoke({"ticket_text": ticket, "messages": []}, config=config)

    snap = graph.get_state(config)
    if snap.next:
        graph.update_state(config, {"admin_decision": "approve", "refund_approved": True})
        result = graph.invoke(None, config=config)

    citations = result.get("citations") or []
    assert citations, "kb_orchestrator must return at least one citation"

    required_fields = {"source", "chunk_id", "start_char", "snippet"}
    for c in citations:
        missing = required_fields - c.keys()
        assert not missing, f"Citation missing fields {missing}: {c}"

    reply = result.get("reply_draft") or ""
    assert "Policy citations:" in reply, (
        f"Expected 'Policy citations:' in reply_draft.\nGot: {reply!r}"
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
