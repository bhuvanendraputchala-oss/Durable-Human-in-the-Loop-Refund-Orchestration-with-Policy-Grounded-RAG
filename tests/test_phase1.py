import os
import json

from app.graph import build_graph


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INTERACTIONS_DIR = os.path.join(ROOT, "interactions")
MOCK_DATA_DIR = os.path.join(ROOT, "mock_data")


def load_json(directory: str, name: str):
    with open(os.path.join(directory, name), "r", encoding="utf-8") as f:
        return json.load(f)

def render_reply(templates: dict, issue_type: str, order: dict) -> str:
    template = templates.get(issue_type)
    if not template:
        return ""
    return (
        template.replace("{{customer_name}}", order.get("customer_name", "Customer"))
        .replace("{{order_id}}", order.get("order_id", ""))
        .strip()
    )

def test_phase1_demo():
    demos = load_json(INTERACTIONS_DIR, "phase1_demo.json")
    orders = load_json(MOCK_DATA_DIR, "orders.json")
    replies = load_json(MOCK_DATA_DIR, "replies.json")

    orders_by_id = {o["order_id"]: o for o in orders}
    templates = {r["issue_type"]: r["template"] for r in replies}

    graph = build_graph()

    for demo in demos:
        conversation_id = demo["conversation_id"]
        expected = demo["expected_outcome"]
        first_user_message = demo["turns"][0]["message"]

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

        expected_issue_type = expected["issue_type"]
        expected_order_id = expected["order_id"]

        assert result_2.get("issue_type") == expected_issue_type, f"{conversation_id} issue_type mismatch"
        assert result_2.get("order_id") == expected_order_id, f"{conversation_id} order_id mismatch"

        expected_order = orders_by_id.get(expected_order_id, {})
        expected_reply = render_reply(templates, expected_issue_type, expected_order)
        actual_reply = (result_2.get("reply_draft") or "").strip()

        assert actual_reply == expected_reply, f"{conversation_id} reply mismatch"


