import os
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

# ----------------------------
# Env
# ----------------------------
CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]

# Default to a model that your /v1/models output shows exists
# (You can override via Render env var CLAUDE_MODEL)
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# Optional shared-secret auth for /analyze
AUTH_TOKEN = os.environ.get("ANALYZER_AUTH_TOKEN", "")

# Regal domains to suppress / identify internal speakers
REGAL_DOMAINS = set(
    d.strip().lower()
    for d in os.environ.get("REGAL_DOMAINS", "regalvoice.com,regal.ai,regal.io").split(",")
    if d.strip()
)

# Optional: pipe-separated list of employee full names for internal speaker detection
# e.g. "Alex Krumm|Maaria Khalid|Courtland Nicholas"
EMPLOYEE_NAMES: List[str] = [
    n.strip() for n in os.environ.get("REGAL_EMPLOYEE_NAMES", "").split("|") if n.strip()
]
EMPLOYEE_NAMES_LOWER = {n.lower() for n in EMPLOYEE_NAMES}

app = FastAPI()


# ----------------------------
# Models
# ----------------------------
class AnalyzeIn(BaseModel):
    clari_call_id: str | None = None
    salesforce_opp_id: str | None = None
    stage_at_time: str | None = None
    segment: str | None = None
    transcript: str = Field(..., min_length=1)


# ----------------------------
# Utils
# ----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def domain_from_email(email: str) -> str:
    email = (email or "").strip()
    if "@" not in email:
        return ""
    return email.split("@", 1)[1].strip().lower()


def display_from_email(email: str) -> str:
    """
    Best-effort display name from email local-part.
    thea.rasmussen@broadrch.com -> thea rasmussen -> Thea Rasmussen
    """
    email = (email or "").strip()
    if "@" not in email:
        return ""
    local = email.split("@", 1)[0]
    local = re.sub(r"[._\-]+", " ", local).strip()
    return " ".join(w.capitalize() for w in local.split() if w)


def looks_like_email(s: str) -> bool:
    s = (s or "").strip()
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s))


def extract_first_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object from a string.
    Handles cases where the model returns extra text before/after the JSON.
    Raises ValueError if no JSON object is found.
    """
    s = (text or "").strip()

    # Fast path: already looks like a standalone JSON object
    if s.startswith("{") and s.endswith("}"):
        return s

    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found in model output")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]

    raise ValueError("Could not find a complete JSON object in model output")


# ----------------------------
# Speaker parsing + normalization
# ----------------------------
def parse_speaker_header(line: str) -> Tuple[str, str, str]:
    """
    Parse a speaker line like:
      "thea.rasmussen@broadrch.com: blah"
      "Christina Scorsis: blah"
      "alex@regalvoice.com: blah"
      "Patrice: blah"

    Returns (display, email, domain)
    - display: human name or email-derived display
    - email: if present, else ""
    - domain: if email present, else ""
    """
    if ":" not in line:
        return "", "", ""

    left = line.split(":", 1)[0].strip()
    if not left:
        return "", "", ""

    # If left is an email
    if looks_like_email(left):
        email = left
        dom = domain_from_email(email)
        display = display_from_email(email) or email
        return display, email, dom

    # Otherwise treat as display name
    display = left
    return display, "", ""


def is_regal_speaker(display: str, email: str, dom: str) -> bool:
    dom = (dom or "").lower()
    if dom and dom in REGAL_DOMAINS:
        return True

    # If email is internal even if dom not parsed
    if email and domain_from_email(email) in REGAL_DOMAINS:
        return True

    # If display matches known employee names (optional)
    if display and display.strip().lower() in EMPLOYEE_NAMES_LOWER:
        return True

    return False


def normalize_speaker_fields(display: str, email: str, dom: str) -> Dict[str, str]:
    """
    Produce stable speaker fields while suppressing Regal.
    speaker_type: "regal" or "prospect"
    speaker_display: best display label
    speaker_email: keep only for non-Regal if present
    speaker_company: for non-Regal, prefer domain if present else "unknown"
    """
    display = (display or "").strip()
    email = (email or "").strip()
    dom = (dom or "").strip().lower()

    regal = is_regal_speaker(display, email, dom)

    speaker_type = "regal" if regal else "prospect"

    # Display preference:
    # - If provided display is an email, convert to pretty name if possible
    # - Else keep as is
    if looks_like_email(display):
        pretty = display_from_email(display)
        speaker_display = pretty or display
    else:
        speaker_display = display or (display_from_email(email) if email else "Unknown")

    # Email + company only for prospects
    speaker_email = "" if regal else (email or "")
    speaker_company = "unknown"
    if not regal:
        if dom:
            speaker_company = dom
        elif speaker_email:
            speaker_company = domain_from_email(speaker_email) or "unknown"

    return {
        "speaker_type": speaker_type,
        "speaker_display": speaker_display,
        "speaker_email": speaker_email,
        "speaker_company": speaker_company,
    }


def enrich_transcript_for_attribution(transcript: str) -> str:
    """
    Rewrite each speaker line to a consistent format that helps the model:
      SPEAKER[prospect|regal] display=<...> email=<...> company=<...>: utterance...

    Enhancements:
    - Speaker identity memory within a single transcript:
        If we see "Patrice" with an email once, later "Patrice:" lines inherit that email/company.
    - Keeps original utterance unchanged.
    - Non-speaker lines pass through unchanged.
    """
    out_lines: List[str] = []

    # In-transcript identity memory: display(lower) -> (email, domain)
    speaker_memory: Dict[str, Tuple[str, str]] = {}

    for raw_line in (transcript or "").splitlines():
        line = raw_line.rstrip("\n")
        display, email, dom = parse_speaker_header(line)
        if not display:
            out_lines.append(raw_line)
            continue

        # Extract utterance safely
        parts = line.split(":", 1)
        if len(parts) < 2:
            out_lines.append(raw_line)
            continue
        utterance = parts[1].strip()

        key = (display or "").strip().lower()

        # If this line includes an email, remember it for this speaker display
        if email:
            speaker_memory[key] = (email, dom or "")

        # If this line has no email, but we've seen this speaker before, inherit it
        elif key in speaker_memory:
            remembered_email, remembered_dom = speaker_memory[key]
            email = remembered_email
            dom = dom or remembered_dom

        meta = normalize_speaker_fields(display, email, dom)

        # Use a stable, parseable marker; keep the original utterance unchanged
        out_lines.append(
            f"SPEAKER[{meta['speaker_type']}] display={meta['speaker_display']} "
            f"email={meta['speaker_email']} company={meta['speaker_company']}: {utterance}"
        )

    return "\n".join(out_lines)


# ----------------------------
# Prompting
# ----------------------------
def build_prompt(payload: AnalyzeIn) -> str:
    enriched = enrich_transcript_for_attribution(payload.transcript)

    return f"""
You are an information extraction engine. You must output VALID JSON ONLY (no markdown, no commentary).

CRITICAL CREDIBILITY RULES:
- Every extracted item MUST include an evidence_quote that is an exact, verbatim substring from the transcript provided (including the SPEAKER[...] prefix).
- If you cannot find a verbatim quote supporting an item, omit that item. Do not infer.
- Do not fabricate names, integrations, compliance requirements, numbers, timelines, outcomes, or next steps.

SPEAKER RULES:
- Only extract items spoken by SPEAKER[prospect]. Ignore SPEAKER[regal].
- Copy speaker fields exactly from the transcript prefix:
  - speaker_display
  - speaker_email
  - speaker_company
  - speaker_type ("prospect" always for extracted items)

Return JSON with this schema (object at top-level):

{{
  "call_id": "{payload.clari_call_id or ""}",
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
- "tags" should include specific systems/terms if present in the prospect's words (salesforce, hubspot, whatsapp, meta, soc2, hipaa, tcpa, data_residency).
- topic_tags should be grounded in the transcript, not guessed.

Inputs:
clari_call_id: {payload.clari_call_id or ""}
salesforce_opp_id: {payload.salesforce_opp_id or ""}
stage_at_time: {payload.stage_at_time or ""}
segment: {payload.segment or ""}

transcript (with SPEAKER prefixes):
{enriched}
""".strip()


# ----------------------------
# Anthropic call
# ----------------------------
def call_claude(prompt: str) -> Tuple[dict, str]:
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

    # Better error propagation
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Anthropic error {r.status_code}: {err}")

    data = r.json()
    content = data.get("content", [])

    # Anthropic returns a list of blocks like [{"type":"text","text":"..."}]
    raw_text = ""
    if isinstance(content, list) and content:
        # Join any text blocks
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                parts.append(block["text"])
        raw_text = "\n".join(parts).strip()

    if not raw_text:
        raise ValueError("Claude response missing text content")

    json_text = extract_first_json_object(raw_text)
    return json.loads(json_text), raw_text


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso(), "model": CLAUDE_MODEL}


@app.post("/analyze")
def analyze(payload: AnalyzeIn, authorization: str | None = Header(default=None)):
    # Optional shared-secret gate
    if AUTH_TOKEN:
        if not authorization or authorization != f"Bearer {AUTH_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    transcript = payload.transcript or ""
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

    raw_text = ""
    try:
        prompt = build_prompt(payload)
        result, raw_text = call_claude(prompt)

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

        # Dedupe evidence quotes while preserving order
        seen = set()
        deduped_quotes = []
        for q in evidence_quotes:
            if q not in seen:
                seen.add(q)
                deduped_quotes.append(q)

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
            "evidence_quotes_json": json.dumps(deduped_quotes[:25]),
        }

    except Exception as e:
        # Return error fields in a way Clay can map + include debug_model_output
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
            "debug_model_output": (raw_text[:4000] if raw_text else ""),
        }
