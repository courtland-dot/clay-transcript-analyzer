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

CLAUDE_TIMEOUT_SECS = int(os.environ.get("CLAUDE_TIMEOUT_SECS", "90"))
CLAUDE_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "3000"))

AUTH_TOKEN = os.environ.get("ANALYZER_AUTH_TOKEN", "")

REGAL_DOMAINS = set(
    d.strip().lower()
    for d in os.environ.get("REGAL_DOMAINS", "regalvoice.com,regal.ai").split(",")
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
    return " ".join(w.capitalize() for w in local.split())

def looks_like_email(s: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", (s or "").strip()))

# ----------------------------
# Speaker parsing
# ----------------------------
def parse_speaker_header(line: str) -> Tuple[str, str, str]:
    if ":" not in line:
        return "", "", ""

    left = line.split(":", 1)[0].strip()

    if looks_like_email(left):
        email = left
        dom = domain_from_email(email)
        display = display_from_email(email)
        return display, email, dom

    return left, "", ""

def is_regal_speaker(display: str, email: str, dom: str) -> bool:
    if dom in REGAL_DOMAINS:
        return True
    if email and domain_from_email(email) in REGAL_DOMAINS:
        return True
    if display and display.lower() in EMPLOYEE_NAMES_LOWER:
        return True
    return False

def normalize_speaker_fields(display: str, email: str, dom: str) -> Dict[str, str]:
    regal = is_regal_speaker(display, email, dom)

    speaker_type = "regal" if regal else "prospect"
    speaker_display = display or display_from_email(email)
    speaker_email = "" if regal else email
    speaker_company = "" if regal else (dom or "unknown")

    return {
        "speaker_type": speaker_type,
        "speaker_display": speaker_display,
        "speaker_email": speaker_email,
        "speaker_company": speaker_company,
    }

# ----------------------------
# Transcript enrichment
# ----------------------------
def enrich_transcript_for_attribution(transcript: str) -> str:

    out_lines: List[str] = []
    speaker_memory: Dict[str, Tuple[str, str]] = {}

    for raw_line in transcript.splitlines():

        display, email, dom = parse_speaker_header(raw_line)

        if not display:
            out_lines.append(raw_line)
            continue

        utterance = raw_line.split(":", 1)[1].strip()

        key = display.lower()

        if email:
            speaker_memory[key] = (email, dom)
        elif key in speaker_memory:
            email, dom = speaker_memory[key]

        meta = normalize_speaker_fields(display, email, dom)

        out_lines.append(
            f"SPEAKER[{meta['speaker_type']}] "
            f"display={meta['speaker_display']} "
            f"email={meta['speaker_email']} "
            f"company={meta['speaker_company']}: {utterance}"
        )

    return "\n".join(out_lines)

# ----------------------------
# Prompt
# ----------------------------
def build_prompt(payload: AnalyzeIn, enriched: str) -> str:

    call_id = payload.clari_call_id or ""

    return f"""
You are an information extraction engine.

Return JSON ONLY.

Only extract items from SPEAKER[prospect].

Return schema:

{{
 "call_id": "{call_id}",
 "questions": [],
 "objections": [],
 "product_feedback": [],
 "buying_signals": [],
 "summary_10_lines": [],
 "topic_tags": []
}}

Transcript:
{enriched}
"""

# ----------------------------
# Claude call
# ----------------------------
def call_claude(prompt: str):

    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": CLAUDE_MAX_TOKENS,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }

    r = requests.post(
        url,
        headers=headers,
        data=json.dumps(body),
        timeout=CLAUDE_TIMEOUT_SECS,
    )

    if r.status_code >= 400:
        raise RuntimeError(r.text)

    data = r.json()

    raw = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            raw += block["text"]

    raw = raw.strip()

    parsed = json.loads(raw)

    return parsed, raw

# ----------------------------
# API
# ----------------------------
@app.post("/analyze")
def analyze(payload: AnalyzeIn, authorization: str | None = Header(default=None)):

    if AUTH_TOKEN:
        if authorization != f"Bearer {AUTH_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    enriched = enrich_transcript_for_attribution(payload.transcript)

    prompt = build_prompt(payload, enriched)

    raw_text = ""

    try:

        result, raw_text = call_claude(prompt)

        questions = result.get("questions", []) or []
        objections = result.get("objections", []) or []
        product_feedback = result.get("product_feedback", []) or []
        buying_signals = result.get("buying_signals", []) or []
        topic_tags = result.get("topic_tags", []) or []
        summary_lines = result.get("summary_10_lines", []) or []

        # ----------------------------
        # Inject call_id for Clay joins
        # ----------------------------
        call_id = payload.clari_call_id or ""

        for arr in (questions, objections, product_feedback, buying_signals):
            for item in arr:
                if isinstance(item, dict):
                    item["call_id"] = call_id

        # Evidence quotes
        evidence_quotes = []

        for q in questions:
            if q.get("evidence_quote"):
                evidence_quotes.append(q["evidence_quote"])

        for o in objections:
            if o.get("evidence_quote"):
                evidence_quotes.append(o["evidence_quote"])

        return {

            "analysis_status": "processed",
            "analysis_error": "",
            "analysis_last_run_at": now_iso(),

            "extraction_json": json.dumps(result),

            "questions_json": json.dumps(questions),
            "objections_json": json.dumps(objections),
            "product_feedback_json": json.dumps(product_feedback),
            "buying_signals_json": json.dumps(buying_signals),

            "summary_lines_json": json.dumps(summary_lines),

            "topic_tags": topic_tags,

            "top_questions_bullets":
                "\n".join(
                    f"- {q.get('normalized') or q.get('verbatim')}"
                    for q in questions[:5]
                ),

            "evidence_quotes_json": json.dumps(evidence_quotes),

        }

    except Exception as e:

        return {
            "analysis_status": "error",
            "analysis_error": str(e),
            "analysis_last_run_at": now_iso(),
            "extraction_json": "",
            "questions_json": "[]",
            "objections_json": "[]",
            "product_feedback_json": "[]",
            "buying_signals_json": "[]",
            "summary_lines_json": "[]",
            "topic_tags": [],
            "top_questions_bullets": "",
            "evidence_quotes_json": "[]",
            "debug_model_output": raw_text,
        }

@app.get("/health")
def health():
    return {"ok": True}
