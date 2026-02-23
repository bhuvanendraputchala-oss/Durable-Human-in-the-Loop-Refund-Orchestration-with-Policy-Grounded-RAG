from __future__ import annotations

import json
import os
from typing import Any, Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOCK_DIR = os.path.join(ROOT, "mock_data")


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


def _load_templates() -> Dict[str, str]:
    replies = load_json("replies.json")
    
    if not isinstance(replies, list):
        raise ValueError("replies.json must contain a list")
    
    templates = {}
    for r in replies:
        if not isinstance(r, dict):
            continue
        issue_type = r.get("issue_type")
        template = r.get("template")
        if issue_type and template:
            templates[issue_type] = str(template)
    
    return templates


TEMPLATES: Dict[str, str] = _load_templates()


def render_reply(issue_type: str, order: Dict[str, Any] | None = None) -> str:
    if order is None:
        order = {}
    
    template = TEMPLATES.get(issue_type)
    if not template:
        return "Hi there, thanks for reaching out. We are looking into your request and will get back to you shortly."

    customer_name = order.get("customer_name") or "there"
    order_id = order.get("order_id") or "your order"
    customer_name = str(customer_name).strip() if customer_name else "there"
    order_id = str(order_id).strip() if order_id else "your order"
    result = template.replace("{{customer_name}}", customer_name)
    result = result.replace("{{order_id}}", order_id)
    
    return result

