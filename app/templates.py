from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_json(filename: str) -> Any:
    path = DATA_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


replies = load_json("replies.json")
TEMPLATES: Dict[str, str] = {r["issue_type"]: r["template"] for r in replies}


def render_reply(issue_type: str, order: Dict[str, Any]) -> str:
    template = TEMPLATES.get(issue_type)
    if not template:
        return "Hi there, thanks for reaching out. We are looking into your request and will get back to you shortly."

    customer_name = order.get("customer_name") or "there"
    order_id = order.get("order_id") or "your order"

    return (
        template.replace("{{customer_name}}", str(customer_name))
        .replace("{{order_id}}", str(order_id))
    )

