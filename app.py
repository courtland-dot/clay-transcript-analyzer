import os
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

# ----------------------------
# Env
# ----------------------------
CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
AUTH_TOKEN = os.environ.get("ANALYZER_AUTH_TOKEN", "")

REGAL_DOMAINS = set(
    d.strip().lower()
    for d in os.environ.get("REGAL_DOMAINS", "regalvoice.com,regal.ai,regal.io").split(",")
    if d.strip()
)

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
    Handles extra text before/after JSON.
    """
    s = (text or "").strip()

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

    # If we get here, we found "{" but never closed it -> most likely truncation
    raise ValueError("Could not find a complete JSON object in model output")

# ----------------------------
# Speaker parsing + normalization
# ----------------------------
def parse_speaker_header(line: str) -> Tuple[str, str, str]:
    if ":" not in line:
        return "", "", ""

    left = line.split(":", 1)[0].strip()
    if not left:
        return "", "", ""

    if looks_like_email(left):
        email = left
        dom = domain_from_email(email)
        display = display_from_email(email) or email
        return display, email, dom

    return left, "", ""

def is_regal_speaker(display: str, email: str, dom: str) -> bool:
    dom = (dom or "").lower()
    if dom and dom in REGAL_DOMAINS:
        return True
    if email and domain_from_email(email) in REGAL_DOMAINS:
        return True
    if display and display.strip().lower() in EMPLOYEE_NAMES_LOWER:
        return True
    return False

def normalize_speaker_fields(display: str, email: str, dom: str) -> Dict[str, str]:
    display = (display or "").strip()
    email = (email or "").strip()
    dom = (dom or "").strip().lower()

    regal = is_regal_speaker(display, email, dom)
    speaker_type = "regal" if regal else "prospect"

    if looks_like_email(display):
        pretty = display_from_email(display)
        speaker_display = pretty or display
    else:
        speaker_display = display or (display_from_email(email) if email else "Unknown")

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
    SPEAKER identity memory inside the same transcript:
    if we see a speaker name with an email once, reuse it later.
    """
    out_lines: List[str] = []
    speaker_memory: Dict[str, Tuple[str, str]] = {}  # display(lower) -> (email, domain)

    for raw_line in (transcript or "").splitlines():
        line = raw_line.rstrip("\n")
        display, email, dom = parse_speaker_header(line)

        if not display:
            out_lines.append(raw_line)
            continue

        parts = line.split(":", 1)
        if len(parts) < 2:
            out_lines.append(raw_line)
            continue
        utterance = parts[1].strip()

        key = display.strip().lower()

        if email:
            speaker_memory[key] = (email, dom or "")
        elif key in speaker_memory:
            remembered_email, remembered_dom = speaker_memory[key]
            email = remembered_email
            dom = dom or remembered_dom

        meta = normalize_speaker_fields(display, email, dom)

        out_lines.append(
            f"SPEAKER[{meta['speaker_type']}] display={meta['speaker_display']} "
            f"email={meta['speaker_email']} company={meta['speaker_company']}: {utterance}"
        )

    return "\n".join(out_lines)

# ----------------------------
# Prompting
# ----------------------------
def build_prompt(payload: AnalyzeIn) -> str:
    call_id = payload.clari_call_id or ""
    sf_opp_id = payload.salesforce_opp_id or ""
    stage = payload.stage_at_time or ""
    segment = payload.segment or ""

    enriched = enrich_transcript_for_attribution(payload.transcript)

    return f"""
You must output VALID JSON ONLY. No markdown. No backticks. No commentary.
Only extract items from lines labeled SPEAKER[prospect]. Ignore SPEAKER[regal] lines entirely.

HARD EVIDENCE RULES (NO EXCEPTIONS):
- Every extracted item MUST include an evidence_quote that is an exact, verbatim substring from the transcript.
- The evidence_quote MUST include the SPEAKER[prospect] prefix verbatim as it appears.
- If you cannot find a verbatim quote supporting an item, OMIT the item.
- Do not infer. Do not guess. Do not fabricate.

OUTPUT SIZE LIMITS (IMPORTANT):
- questions: max 12
- objections: max 12
- product_feedback: max 12
- buying_signals: max 12
- summary_10_lines: max 10 (return fewer if you can’t ground them)
- topic_tags: max 15
- topic_tag_evidence: max 25

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
  "summary_10_lines": [
    {{
      "line": "",
      "evidence_quote": ""
    }}
  ],
  "topic_tags": ["..."],
  "topic_tag_evidence": [
    {{
      "topic_tag": "",
      "evidence_quote": ""
    }}
  ],
  "quality_flags": {{
    "transcript_too_short": false,
    "low_signal": false
  }}
}}

Rules:
- Output JSON only.
- Keep verbatim fields short (<= 240 chars).
- normalized should be a reusable FAQ/blog title.
- tags/topic_tags must be grounded (no guessing).
- topic_tags must be lowercase snake_case.

Inputs:
clari_call_id: {call_id}
salesforce_opp_id: {sf_opp_id}
stage_at_time: {stage}
segment: {segment}

transcript:
{enriched}
""".strip()

# ----------------------------
# Anthropic call (RAW TEXT ONLY)
# ----------------------------
def call_claude_raw(prompt: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": CLAUDE_MODEL,
        # Bump this to reduce truncation risk
        "max_tokens": 5000,
        "temperature": 0,
        "system": "Return only a single valid JSON object. Do not include any other text.",
        "messages": [{"role": "user", "content": prompt}],
    }

    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=120)

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Anthropic error {r.status_code}: {err}")

    data = r.json()
    content = data.get("content", [])

    if not isinstance(content, list) or not content:
        return ""

    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
            parts.append(block["text"])

    return "\n".join(parts).strip()

# ----------------------------
# Post-processing helpers
# ----------------------------
def coerce_summary_lines(summary_val) -> Tuple[List[str], List[str]]:
    lines: List[str] = []
    evs: List[str] = []
    if isinstance(summary_val, list):
        for item in summary_val:
            if isinstance(item, dict):
                line = (item.get("line") or "").strip()
                ev = (item.get("evidence_quote") or "").strip()
                if line:
                    lines.append(line)
                if ev:
                    evs.append(ev)
            elif isinstance(item, str):
                s = item.strip()
                if s:
                    lines.append(s)
    return lines, evs

def coerce_topic_tag_evidence(val) -> List[str]:
    evs: List[str] = []
    if isinstance(val, list):
        for item in val:
            if isinstance(item, dict):
                ev = (item.get("evidence_quote") or "").strip()
                if ev:
                    evs.append(ev)
    return evs

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso(), "model": CLAUDE_MODEL}

@app.post("/analyze")
def analyze(payload: AnalyzeIn, authorization: str | None = Header(default=None)):
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
            "topic_tag_evidence": [],
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
            "summary_lines_json": json.dumps([]),
            "topic_tag_evidence_json": json.dumps([]),
        }

    raw_text = ""
    try:
        prompt = build_prompt(payload)
        raw_text = call_claude_raw(prompt)

        if not raw_text:
            raise ValueError("Claude returned empty text content")

        json_text = extract_first_json_object(raw_text)
        result = json.loads(json_text)

        questions = result.get("questions", []) or []
        objections = result.get("objections", []) or []
        product_feedback = result.get("product_feedback", []) or []
        buying_signals = result.get("buying_signals", []) or []
        topic_tags = result.get("topic_tags", []) or []

        summary_lines, summary_evidence = coerce_summary_lines(result.get("summary_10_lines", []))
        topic_tag_evidence_quotes = coerce_topic_tag_evidence(result.get("topic_tag_evidence", []))

        evidence_quotes: List[str] = []
        for q in questions[:25]:
            ev = q.get("evidence_quote")
            if ev:
                evidence_quotes.append(ev)
        for o in objections[:25]:
            ev = o.get("evidence_quote")
            if ev:
                evidence_quotes.append(ev)

        evidence_quotes.extend(summary_evidence[:25])
        evidence_quotes.extend(topic_tag_evidence_quotes[:25])

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
            "summary_lines_json": json.dumps(summary_lines),
            "topic_tag_evidence_json": json.dumps(topic_tag_evidence_quotes),
            # Helpful for Clay debugging when needed:
            "debug_model_output": raw_text[:4000],
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
            "summary_lines_json": json.dumps([]),
            "topic_tag_evidence_json": json.dumps([]),
            # NOW this will actually show what Claude returned (even if invalid JSON)
            "debug_model_output": (raw_text[:4000] if raw_text else ""),
        }
