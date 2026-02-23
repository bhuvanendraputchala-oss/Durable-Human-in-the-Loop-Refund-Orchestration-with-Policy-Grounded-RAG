from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Any, Dict


def refund_preview(order_id: str, amount: float) -> Dict[str, Any]:
    return {
        "preview_id": f"prev_{uuid.uuid4().hex[:8]}",
        "order_id": order_id,
        "amount": round(float(amount), 2),
        "method": "original_payment_method",
        "eta": (date.today() + timedelta(days=5)).isoformat(),
        "status": "preview",
    }


def refund_commit(order_id: str, preview_id: str) -> Dict[str, Any]:
    txn_seed = preview_id.replace("prev_", "")
    return {
        "transaction_id": f"txn_{txn_seed}",
        "preview_id": preview_id,
        "order_id": order_id,
        "status": "committed",
        "committed_at": date.today().isoformat(),
    }
