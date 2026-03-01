import os
import json
from datetime import datetime, timezone
import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

# --- Env ---
CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
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
- \"normalized\" should be reusable as a FAQ/blog title.
- \"tags\" should include specific systems/terms if present (salesforce, hubspot, whatsapp, meta, soc2, hipaa, tcpa, data_residency).
- Set temperature to 0 (handled by caller) and do not add creative content.

Inputs:
clari_call_id: {payload.clari_call_id or ""}
salesforce_opp_id: {payload.salesforce_opp_id or ""}
stage_at_time: {payload.stage_at_time or ""}
segment: {payload.segment or ""}

transcript:
{payload.transcript}
""".strip()


def call_claude(prompt: str) -> dict:
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
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=120)
    r.raise_for_status()

    content = r.json().get("content", [])
    if not content or "text" not in content[0]:
        raise ValueError("Claude response missing content text")

    raw_text = content[0]["text"].strip()

    # Sometimes models wrap JSON in whitespace; we still require strict parse.
    return json.loads(raw_text)


@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso()}


@app.post("/analyze")
def analyze(payload: AnalyzeIn, authorization: str | None = Header(default=None)):
    # Optional shared-secret gate
    if AUTH_TOKEN:
        if not authorization or authorization != f"Bearer {AUTH_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    transcript = payload.transcript or ""
    if len(transcript) < 400:
        # Return column-friendly flat fields for Clay mapping
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

        evidence_quotes = []
        for q in questions[:25]:
            if q.get("evidence_quote"):
                evidence_quotes.append(q["evidence_quote"])
        for o in objections[:25]:
            if o.get("evidence_quote"):
                evidence_quotes.append(o["evidence_quote"])

        top_bullets = []
        for q in questions[:7]:
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
        # Return error fields in a way Clay can map
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
