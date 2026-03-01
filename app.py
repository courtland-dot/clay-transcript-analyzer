import os
import json
import re
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

# =========================
# ENV
# =========================
CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
AUTH_TOKEN = os.environ.get("ANALYZER_AUTH_TOKEN", "")

# Comma-separated list in Render env var: "First Last,Another Name,..."
REGAL_EMPLOYEE_NAMES = set(
    n.strip().lower()
    for n in os.environ.get("REGAL_EMPLOYEE_NAMES", "").split(",")
    if n.strip()
)

# Comma-separated domains: "regalvoice.com,regal.ai,regal.io"
REGAL_DOMAINS = [
    d.strip().lower()
    for d in os.environ.get("REGAL_DOMAINS", "").split(",")
    if d.strip()
]

app = FastAPI()

# =========================
# MODELS
# =========================
class AnalyzeIn(BaseModel):
    clari_call_id: str | None = None
    salesforce_opp_id: str | None = None
    stage_at_time: str | None = None
    segment: str | None = None
    transcript: str = Field(..., min_length=1)


# =========================
# HELPERS
# =========================
EMAIL_REGEX = re.compile(r'([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})', re.IGNORECASE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(s: str) -> str:
    return (s or "").strip()


def classify_regal(name_display: Optional[str], email: Optional[str]) -> bool:
    """Return True if speaker belongs to Regal (by domain or by employee name)."""
    if email:
        m = EMAIL_REGEX.search(email)
        if m:
            domain = m.group(2).lower()
            if domain in REGAL_DOMAINS:
                return True
    if name_display and name_display.strip().lower() in REGAL_EMPLOYEE_NAMES:
        return True
    return False


def parse_speaker_header(line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse a transcript line like:
      "Full Name: blah"
      "email@domain.com: blah"
      "Full Name <email@domain.com>: blah" (rare)
    Returns: (speaker_display, speaker_email, speaker_company_domain)
    """
    if ":" not in line:
        return None, None, None

    header = line.split(":", 1)[0].strip()
    if not header:
        return None, None, None

    # If header contains an email, extract it
    m = EMAIL_REGEX.search(header)
    if m:
        email = m.group(0)
        domain = m.group(2).lower()
        # If there is a name around the email, use it as display; else use email local part
        display = header.replace(email, "").strip(" <>-—–\t")
        if not display:
            display = email.split("@", 1)[0]
        return display, email, domain

    # No email in header => it's probably a name
    return header, None, None


def normalize_speaker_fields(display: Optional[str], email: Optional[str], company_domain: Optional[str]) -> Dict[str, Any]:
    """
    Return consistent speaker fields.
    speaker_company is the domain if present; otherwise "unknown".
    speaker_type is "regal" or "prospect".
    """
    display = _clean(display) or "Unknown"
    email = _clean(email) or ""
    company_domain = _clean(company_domain) or "unknown"

    is_regal = classify_regal(display if display != "Unknown" else None, email if email else None)
    speaker_type = "regal" if is_regal else "prospect"

    # If email is present and domain is a Regal domain, suppress company label to avoid leaking internal domains
    speaker_company = company_domain
    if speaker_type == "regal":
        speaker_company = "Regal"

    return {
        "speaker_display": display,
        "speaker_email": email,
        "speaker_company": speaker_company,
        "speaker_type": speaker_type,
    }


def enrich_transcript_for_attribution(transcript: str) -> str:
    """
    Rewrite each speaker line to a consistent format that helps the model:
      SPEAKER[prospect|regal] display=<...> email=<...> company=<...>: utterance...
    Non-speaker lines pass through unchanged.
    """
    out_lines: List[str] = []
    for raw_line in transcript.splitlines():
        line = raw_line.rstrip("\n")
        display, email, dom = parse_speaker_header(line)
        if not display:
            out_lines.append(raw_line)
            continue

        utterance = line.split(":", 1)[1].strip()
        meta = normalize_speaker_fields(display, email, dom)

        # Use a stable, parseable marker; keep the original utterance unchanged
        out_lines.append(
            f"SPEAKER[{meta['speaker_type']}] display={meta['speaker_display']} email={meta['speaker_email']} company={meta['speaker_company']}: {utterance}"
        )
    return "\n".join(out_lines)


# =========================
# PROMPT
# =========================
def build_prompt(payload: AnalyzeIn, enriched_transcript: str) -> str:
    """
    Full production schema. Evidence quotes MUST be exact substrings from the transcript.
    Only extract PROSPECT speaker statements (ignore SPEAKER[regal] lines).
    """
    call_id = payload.clari_call_id or ""
    sf_opp_id = payload.salesforce_opp_id or ""
    stage = payload.stage_at_time or ""
    segment = payload.segment or ""

    return f"""
You are an information extraction engine. You must output VALID JSON ONLY (no markdown, no backticks, no commentary).
Only extract items from lines labeled SPEAKER[prospect]. Ignore SPEAKER[regal] lines.

Every extracted item MUST include an evidence_quote that is an exact, verbatim substring from the transcript.
The evidence_quote MUST include the SPEAKER[...] prefix line content verbatim as it appears.
If you cannot find a verbatim quote supporting an item, omit that item.
Do not infer. Do not guess. Do not fabricate names, integrations, compliance requirements, numbers, or outcomes.
Return empty arrays when nothing is found.

Extract structured buyer questions, objections, product feedback moments, and buying signals from this call transcript.

Return JSON with this schema:

{{
  "call_id": "{call_id}",
  "questions": [
    {{
      "verbatim": "",
      "speaker_display": "",
      "speaker_email": "",
      "speaker_company": "",
      "speaker_type": "prospect",
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
      "speaker_display": "",
      "speaker_email": "",
      "speaker_company": "",
      "speaker_type": "prospect",
      "category": "integration|security|pricing|implementation|product|roi|timeline|other",
      "evidence_quote": "",
      "confidence": 0.0
    }}
  ],
  "product_feedback": [
    {{
      "type": "feature_request|bug|confusion|missing_capability|competitor_comparison|implementation_friction",
      "verbatim": "",
      "speaker_display": "",
      "speaker_email": "",
      "speaker_company": "",
      "speaker_type": "prospect",
      "evidence_quote": ""
    }}
  ],
  "buying_signals": [
    {{
      "type": "exec_sponsorship|clear_success_criteria|urgency|strong_value_alignment|procurement_motion|next_steps_committed",
      "verbatim": "",
      "speaker_display": "",
      "speaker_email": "",
      "speaker_company": "",
      "speaker_type": "prospect",
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
- Set confidence 0.0 to 1.0, conservative.
- Do not add creative content.

Inputs:
clari_call_id: {call_id}
salesforce_opp_id: {sf_opp_id}
stage_at_time: {stage}
segment: {segment}

transcript:
{enriched_transcript}
""".strip()


# =========================
# CLAUDE CALL + ROBUST JSON PARSE
# =========================
def _extract_text_from_anthropic(payload: Dict[str, Any]) -> str:
    blocks = payload.get("content", []) or []
    text_parts: List[str] = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            text_parts.append(b.get("text", ""))
    return "".join(text_parts).strip()


def _safe_json_loads(raw: str) -> Any:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Claude returned empty text")

    # 1) Strict parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2) Fallback: attempt to locate first JSON object/array inside the text
    start_obj = raw.find("{")
    start_arr = raw.find("[")
    starts = [i for i in (start_obj, start_arr) if i != -1]
    if not starts:
        raise ValueError(f"Claude returned non-JSON text (first 300 chars): {raw[:300]}")

    start = min(starts)
    trimmed = raw[start:].strip()

    end_obj = trimmed.rfind("}")
    end_arr = trimmed.rfind("]")
    end = max(end_obj, end_arr)
    if end == -1:
        raise ValueError(f"Claude returned incomplete JSON (first 300 chars): {trimmed[:300]}")

    candidate = trimmed[: end + 1]
    return json.loads(candidate)


def call_claude(prompt: str) -> Dict[str, Any]:
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
    if r.status_code >= 400:
        # keep this concise for Clay mapping
        raise Exception(f"Anthropic error {r.status_code}: {r.text[:2000]}")

    data = r.json()
    raw_text = _extract_text_from_anthropic(data)
    parsed = _safe_json_loads(raw_text)

    if not isinstance(parsed, dict):
        raise ValueError("Claude returned JSON but not an object")
    return parsed


# =========================
# ROUTES
# =========================
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

    # Short transcript fast-path
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
        enriched = enrich_transcript_for_attribution(transcript)
        prompt = build_prompt(payload, enriched)
        result = call_claude(prompt)

        questions = result.get("questions", []) or []
        objections = result.get("objections", []) or []
        product_feedback = result.get("product_feedback", []) or []
        buying_signals = result.get("buying_signals", []) or []
        topic_tags = result.get("topic_tags", []) or []

        # Evidence quotes quick list (Clay-friendly)
        evidence_quotes: List[str] = []
        for q in questions[:25]:
            if isinstance(q, dict) and q.get("evidence_quote"):
                evidence_quotes.append(q["evidence_quote"])
        for o in objections[:25]:
            if isinstance(o, dict) and o.get("evidence_quote"):
                evidence_quotes.append(o["evidence_quote"])
        for pf in product_feedback[:25]:
            if isinstance(pf, dict) and pf.get("evidence_quote"):
                evidence_quotes.append(pf["evidence_quote"])
        for bs in buying_signals[:25]:
            if isinstance(bs, dict) and bs.get("evidence_quote"):
                evidence_quotes.append(bs["evidence_quote"])

        # Top normalized questions bullets
        top_bullets: List[str] = []
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
