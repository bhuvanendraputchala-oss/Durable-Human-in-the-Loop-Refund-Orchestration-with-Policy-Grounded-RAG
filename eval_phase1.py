import json
import os

from dotenv import load_dotenv
from langfuse import Langfuse

from app.graph import build_graph


load_dotenv()

ROOT = os.path.abspath(os.path.dirname(__file__))
INTERACTIONS_DIR = os.path.join(ROOT, "interactions")
MOCK_DATA_DIR = os.path.join(ROOT, "mock_data")


def load_json(directory: str, name: str):
    path = os.path.join(directory, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

demos = load_json(INTERACTIONS_DIR, "phase1_demo.json")
orders = load_json(MOCK_DATA_DIR, "orders.json")
replies = load_json(MOCK_DATA_DIR, "replies.json")


ORDERS_BY_ID = {o["order_id"]: o for o in orders}
TEMPLATES = {r["issue_type"]: r["template"] for r in replies}


def render_reply(issue_type: str, order: dict) -> str:
    template = TEMPLATES.get(issue_type)
    if not template:
        return ""
    return (
        template.replace("{{customer_name}}", order.get("customer_name", "Customer"))
        .replace("{{order_id}}", order.get("order_id", ""))
        .strip()
    )


langfuse = Langfuse()
graph = build_graph()

for demo in demos:
    conversation_id = demo.get("conversation_id")
    expected = demo.get("expected_outcome", {})
    turns = demo.get("turns", [])

    if not turns:
        print(f"{conversation_id} skipped, no turns found")
        continue

    first_user_message = turns[0].get("message", "")

    initial_state = {
        "ticket_text": first_user_message,
        "messages": [],
    }

    result_1 = graph.invoke(initial_state)

    result_2 = graph.invoke(
        {
            **result_1,
            "admin_decision": "approve",
            "admin_notes": "ok",
        }
    )

    expected_issue_type = expected.get("issue_type")
    expected_order_id = expected.get("order_id")

    issue_type_match = int(result_2.get("issue_type") == expected_issue_type)
    order_id_match = int(result_2.get("order_id") == expected_order_id)

    expected_order = ORDERS_BY_ID.get(expected_order_id, {})
    expected_reply = render_reply(expected_issue_type or "", expected_order)
    actual_reply = (result_2.get("reply_draft") or "").strip()
    reply_template_match = int(actual_reply == expected_reply)

    trace = langfuse.trace(
        name="phase1_eval",
        metadata={
            "conversation_id": conversation_id,
            "expected_issue_type": expected_issue_type,
            "expected_order_id": expected_order_id,
        },
        input=initial_state,
        output=result_2,
    )

    langfuse.score(trace_id=trace.id, name="issue_type_match", value=issue_type_match)
    langfuse.score(trace_id=trace.id, name="order_id_match", value=order_id_match)
    langfuse.score(
        trace_id=trace.id,
        name="reply_template_match",
        value=reply_template_match,
    )

    print(
        f"{conversation_id} "
        f"issue_type_match={issue_type_match} "
        f"order_id_match={order_id_match} "
        f"reply_template_match={reply_template_match}"
    )

print("Done")
