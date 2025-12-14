from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional


class TriageState(TypedDict, total=False):
    """State schema for the triage graph."""
    messages: List[Dict[str, str]]
    ticket_text: str
    evidence: Dict[str, Any]
    order_id: Optional[str]
    issue_type: Optional[str]
    recommendation: Optional[str]
    needs_admin: Optional[bool]
    admin_decision: Optional[str]
    admin_notes: Optional[str]
    reply_draft: Optional[str]
