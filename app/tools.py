"""Bounded tool / action (Tier B): order-status lookup.

This is the single bounded action from the design. It is intentionally narrow:
  * input is validated against a strict order-id pattern (guardrail),
  * it only reads from a fixed orders fixture (no writes, no PII beyond status),
  * it is traced with @observe so the tool call shows up in Langfuse.

The agent router (see app/agent.py) only invokes it when a question clearly
asks about a specific order id; everything else goes to RAG.
"""
from __future__ import annotations

import re

from .observability import observe

ORDER_ID_RE = re.compile(r"\bNW-(\d{4})\b", re.IGNORECASE)

# Fixed fixture standing in for an orders service / DB.
_ORDERS: dict[str, dict] = {
    "NW-1001": {"status": "Shipped", "carrier": "UPS", "eta": "2 business days"},
    "NW-1002": {"status": "Processing", "carrier": None, "eta": "not yet shipped"},
    "NW-1003": {"status": "Delivered", "carrier": "FedEx", "eta": "delivered"},
    "NW-1004": {"status": "Cancelled", "carrier": None, "eta": "n/a"},
}


def extract_order_id(text: str) -> str | None:
    m = ORDER_ID_RE.search(text or "")
    return f"NW-{m.group(1)}" if m else None


@observe(name="tool_order_status")
def lookup_order_status(order_id: str) -> dict:
    """Bounded, read-only order-status lookup with input validation."""
    if not order_id or not ORDER_ID_RE.fullmatch(order_id):
        return {"ok": False, "error": "invalid_order_id",
                "message": "Order id must look like NW-1234."}
    order = _ORDERS.get(order_id.upper())
    if not order:
        return {"ok": False, "error": "not_found",
                "message": f"No order found for {order_id}."}
    return {"ok": True, "order_id": order_id.upper(), **order}


def format_order_answer(result: dict) -> str:
    if not result.get("ok"):
        return result.get("message", "I couldn't look that order up.")
    carrier = f" via {result['carrier']}" if result.get("carrier") else ""
    return (f"Order {result['order_id']} is **{result['status']}**{carrier} "
            f"(ETA: {result['eta']}).")
 