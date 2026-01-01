"""JSONL analytics endpoints.

Frontend wiring (optional, API_BASE already includes /lexi):

import { apiFetch } from "../services/api";

const storageKey = "lexi_visitor_id";
let visitorId = window.localStorage.getItem(storageKey);
if (!visitorId) {
  visitorId = crypto.randomUUID();
  window.localStorage.setItem(storageKey, visitorId);
}
apiFetch("/analytics/visit", {
  method: "POST",
  body: JSON.stringify({ visitor_id: visitorId }),
});
setInterval(() => {
  apiFetch("/analytics/heartbeat", {
    method: "POST",
    body: JSON.stringify({ visitor_id: visitorId }),
  });
}, 30_000);

CURL smoke test:
export BASE="https://api.lexicompanion.com"
VID="test_$(date +%s)"

curl -sS -X POST "$BASE/lexi/analytics/visit" -H "Content-Type: application/json" -d "{\"visitor_id\":\"$VID\"}" | jq .
curl -sS -X POST "$BASE/lexi/analytics/heartbeat" -H "Content-Type: application/json" -d "{\"visitor_id\":\"$VID\"}" | jq .
curl -sS "$BASE/lexi/analytics/summary" | jq .
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..analytics.jsonl_analytics import get_summary, record_heartbeat

router = APIRouter(prefix="/lexi/analytics", tags=["analytics"])


class AnalyticsPayload(BaseModel):
    visitor_id: str | None = None


def _require_visitor_id(payload: AnalyticsPayload) -> str:
    visitor_id = (payload.visitor_id or "").strip()
    if not visitor_id:
        raise HTTPException(status_code=400, detail="visitor_id is required")
    return visitor_id


@router.post("/visit")
def analytics_visit(payload: AnalyticsPayload) -> dict[str, object]:
    visitor_id = _require_visitor_id(payload)
    return record_heartbeat(visitor_id)


@router.post("/heartbeat")
def analytics_heartbeat(payload: AnalyticsPayload) -> dict[str, object]:
    visitor_id = _require_visitor_id(payload)
    return record_heartbeat(visitor_id)


@router.get("/summary")
def analytics_summary() -> dict[str, object]:
    return get_summary()
