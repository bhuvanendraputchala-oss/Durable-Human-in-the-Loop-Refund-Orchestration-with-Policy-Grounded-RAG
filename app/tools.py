from __future__ import annotations
import json
import os
from typing import Dict, Any
from langchain_core.tools import tool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOCK_DIR = os.path.join(ROOT, "mock_data")


def load(name):
    path = os.path.join(MOCK_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {name}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error reading {name}: {e}") from e

ORDERS = load("orders.json")
ISSUES = load("issues.json")
REPLIES = load("replies.json")

ORDER_ID_TO_ORDER: Dict[str, Dict[str, Any]] = {o["order_id"]: o for o in ORDERS}

@tool
def fetch_order(order_id: str) -> Dict[str, Any]:
    """
    Fetch an order record by order_id from orders.json.
    Returns a small payload that is safe to store in evidence.
    """
    order = ORDER_ID_TO_ORDER.get(order_id)
    if order is None:
        return {"found": False, "order_id": order_id}

    return {"found": True, "order": order}