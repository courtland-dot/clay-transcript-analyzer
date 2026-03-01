import os
import json
from datetime import datetime, timezone
import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

# --- Env ---
CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]
# Use an alias by default so you don't get stuck on a dated model string
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
AUTH_TOKEN = os.environ.get("ANALYZER_AUTH_TOKEN", "")

app = FastAPI()


class AnalyzeIn(BaseModel):
    clari_call_id: str | None = None
    salesforce_opp_id: str | None = None
    stage_at_time: str | None = None
    segment: str | None = None
    transcript: str = Field(..., min_length=1)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_prompt(payload: AnalyzeIn) -> str:
    return f"""
You are an information extraction engine. You must output VALID JSON ONLY (no markdown, no commentary).
Every extracted item MUST include an evidence_quote that is an exact, verbatim substring from the transcript.
If you cannot find a verbatim quote supporting an item, omit that item.
Do not infer. Do not guess. Do not fabricate names, integrations, compliance requirements, numbers, or outcomes.
Return empty arrays when nothing is found.

Extract structured buyer questions, objections, product feedback moments, and buying signals from this call transcript.

Return JSON with this schema:

{{
  "call_id": "{payload.clari_call_id or ""}",
  "questions": [
    {{
      "verbatim": "",
      "normalized": "",
      "category": "integration|security|pricing|implementation|product|roi|timeline|other",
      "tags": ["..."],
      "evidence_quote": "",
      "confidence": 0.0
    }}
  ],
  "objections": [
    {{
      "verbatim": "",
      "category": "integration|security|pricing|implementation|product|roi|timeline|other",
      "evidence_quote": "",
      "confidence": 0.0
    }}
  ],
  "product_feedback": [
    {{
      "type": "feature_request|bug|confusion|missing_capability|competitor_comparison|implementation_friction",
      "verbatim": "",
      "evidence_quote": ""
    }}
  ],
  "buying_signals": [
    {{
      "type": "exec_sponsorship|clear_success_criteria|urgency|strong_value_alignment|procurement_motion|next_steps_committed",
      "verbatim": "",
      "evidence_quote": "",
      "confidence": 0.0
    }}
  ],
  "summary_10_lines": ["..."],
  "topic_tags": ["..."],
  "quality_flags": {{
    "transcript_too_short": false,
    "low_signal": false
  }}
}}

Rules:
- Output JSON only.
- Keep verbatim fields short (<= 240 chars).
- "normalized" should be reusable as a FAQ/blog title.
- "tags" should include specific systems/terms if present (salesforce, hubspot, whatsapp, meta, soc2, hipaa, tcpa, data_residency).
- Do not add creative content.

Inputs:
clari_call_id: {payload.clari_call_id or ""}
salesforce_opp_id: {payload.salesforce_opp_id or ""}
stage_at_time: {payload.stage_at_time or ""}
segment: {payload.segment or ""}

transcript:
{payload.transcript}
""".strip()


def call_claude(prompt: str) -> dict:
    """
    Calls Anthropic Messages API and returns parsed JSON.
    Includes defensive error reporting (status + body) and robust JSON extraction fallback.
    """
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": 3000,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }

    r = requests.post(url, headers=headers, json=body, timeout=120)

    # Provide the actual Anthropic error body so debugging is instant (invalid key/model/etc.)
    if r.status_code >= 400:
        raise ValueError(f"Anthropic error {r.status_code}: {r.text[:2000]}")

    data = r.json()

    # Messages API returns content blocks; concatenate text blocks
    content_blocks = data.get("content", []) or []
    text_parts: list[str] = []
    for b in content_blocks:
        if isinstance(b, dict) and b.get("type") == "text" and "text" in b:
            text_parts.append(b["text"])

    raw_text = "".join(text_parts).strip()
    if not raw_text:
        raise ValueError(f"Claude response missing text content: {json.dumps(data)[:2000]}")

    # Strict parse first
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: extract the first JSON object/array from the response
        start_obj = raw_text.find("{")
        start_arr = raw_text.find("[")
        starts = [i for i in (start_obj, start_arr) if i != -1]
        if not starts:
            raise ValueError(f"Claude returned non-JSON text (first 500 chars): {raw_text[:500]}")

        start = min(starts)
        trimmed = raw_text[start:].strip()

        # Attempt to trim trailing junk by finding the last closing brace/bracket
        end_obj = trimmed.rfind("}")
        end_arr = trimmed.rfind("]")
        end = max(end_obj, end_arr)
        if end == -1:
            raise ValueError(f"Claude returned incomplete JSON (first 500 chars): {trimmed[:500]}")

        candidate = trimmed[: end + 1]
        return json.loads(candidate)


@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso()}


@app.post("/analyze")
def analyze(payload: AnalyzeIn, authorization: str | None = Header(default=None)):
    # Optional shared-secret gate
    if AUTH_TOKEN:
        expected = f"bearer {AUTH_TOKEN}".strip().lower()
        got = (authorization or "").strip().lower()
        if got != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    transcript = payload.transcript or ""

    # Short transcript guard (keeps costs low and avoids low-signal junk)
    if len(transcript) < 400:
        base = {
            "call_id": payload.clari_call_id or "",
            "questions": [],
            "objections": [],
            "product_feedback": [],
            "buying_signals": [],
            "summary_10_lines": [],
            "topic_tags": [],
            "quality_flags": {"transcript_too_short": True, "low_signal": True},
        }
        return {
            "analysis_status": "processed",
            "analysis_error": "",
            "analysis_last_run_at": now_iso(),
            "extraction_json": json.dumps(base),
            "questions_json": json.dumps([]),
            "objections_json": json.dumps([]),
            "product_feedback_json": json.dumps([]),
            "buying_signals_json": json.dumps([]),
            "topic_tags": [],
            "top_questions_bullets": "",
            "evidence_quotes_json": json.dumps([]),
        }

    try:
        prompt = build_prompt(payload)
        result = call_claude(prompt)

        questions = result.get("questions", []) or []
        objections = result.get("objections", []) or []
        product_feedback = result.get("product_feedback", []) or []
        buying_signals = result.get("buying_signals", []) or []
        topic_tags = result.get("topic_tags", []) or []

        # Gather a compact set of receipts for quick human validation
        evidence_quotes: list[str] = []
        for q in questions[:25]:
            if isinstance(q, dict) and q.get("evidence_quote"):
                evidence_quotes.append(q["evidence_quote"])
        for o in objections[:25]:
            if isinstance(o, dict) and o.get("evidence_quote"):
                evidence_quotes.append(o["evidence_quote"])

        # Friendly bullets for quick scanning
        top_bullets: list[str] = []
        for q in questions[:7]:
            if not isinstance(q, dict):
                continue
            norm = q.get("normalized") or q.get("verbatim")
            if norm:
                top_bullets.append(f"- {norm}")

        return {
            "analysis_status": "processed",
            "analysis_error": "",
            "analysis_last_run_at": now_iso(),
            "extraction_json": json.dumps(result),
            "questions_json": json.dumps(questions),
            "objections_json": json.dumps(objections),
            "product_feedback_json": json.dumps(product_feedback),
            "buying_signals_json": json.dumps(buying_signals),
            "topic_tags": topic_tags,
            "top_questions_bullets": "\n".join(top_bullets),
            "evidence_quotes_json": json.dumps(evidence_quotes[:25]),
        }

    except Exception as e:
        return {
            "analysis_status": "error",
            "analysis_error": str(e)[:4000],
            "analysis_last_run_at": now_iso(),
            "extraction_json": "",
            "questions_json": json.dumps([]),
            "objections_json": json.dumps([]),
            "product_feedback_json": json.dumps([]),
            "buying_signals_json": json.dumps([]),
            "topic_tags": [],
            "top_questions_bullets": "",
            "evidence_quotes_json": json.dumps([]),
        }
