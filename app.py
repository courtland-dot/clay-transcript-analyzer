import os
import json
import re
from datetime import datetime, timezone
import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

# =========================
# ENV
# =========================

CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
AUTH_TOKEN = os.environ.get("ANALYZER_AUTH_TOKEN", "")

REGAL_EMPLOYEE_NAMES = set(
    n.strip().lower()
    for n in os.environ.get("REGAL_EMPLOYEE_NAMES", "").split(",")
    if n.strip()
)

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
    transcript: str = Field(..., min_length=1)

# =========================
# HELPERS
# =========================

def now_iso():
    return datetime.now(timezone.utc).isoformat()


EMAIL_REGEX = r'([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})'


def parse_speaker(raw_line: str):

    match = re.match(r"^(.*?):", raw_line)
    if not match:
        return None, None, None

    speaker_raw = match.group(1).strip()

    email_match = re.search(EMAIL_REGEX, speaker_raw)

    if email_match:
        email = email_match.group(0)
        domain = email_match.group(2).lower()

        name_guess = speaker_raw.replace(email, "").strip()

        speaker_display = name_guess if name_guess else email.split("@")[0]
        speaker_company = domain

        if (
            domain in REGAL_DOMAINS
            or speaker_display.lower() in REGAL_EMPLOYEE_NAMES
        ):
            speaker_type = "regal"
        else:
            speaker_type = "prospect"

        return speaker_display, email, speaker_type

    name = speaker_raw.strip()

    if name.lower() in REGAL_EMPLOYEE_NAMES:
        return name, None, "regal"

    return name, None, "prospect"


def enrich_transcript(transcript: str):

    enriched_lines = []

    for line in transcript.splitlines():

        speaker, email, speaker_type = parse_speaker(line)

        if not speaker:
            enriched_lines.append(line)
            continue

        prefix = f"{speaker_type.upper()}[{speaker}]"
        enriched_lines.append(f"{prefix}: {line.split(':',1)[1].strip()}")

    return "\n".join(enriched_lines)


# =========================
# PROMPT
# =========================

def build_prompt(call_id, transcript):

    return f"""
You are an information extraction engine.

Only extract statements spoken by PROSPECT speakers.
Ignore REGAL speakers.

Return VALID JSON ONLY.

Transcript:
{transcript}

Return schema:

{{
 "call_id":"{call_id}",
 "questions":[],
 "objections":[],
 "product_feedback":[],
 "buying_signals":[],
 "summary_10_lines":[],
 "topic_tags":[],
 "quality_flags":{{
   "transcript_too_short":false,
   "low_signal":false
 }}
}}
""".strip()


# =========================
# CLAUDE CALL
# =========================

def call_claude(prompt):

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

    if r.status_code != 200:
        raise Exception(f"Anthropic error {r.status_code}: {r.text}")

    content = r.json()["content"][0]["text"]

    return json.loads(content)


# =========================
# ROUTES
# =========================

@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso()}


@app.post("/analyze")
def analyze(payload: AnalyzeIn, authorization: str | None = Header(default=None)):

    if AUTH_TOKEN:
        if authorization != f"Bearer {AUTH_TOKEN}":
            raise HTTPException(status_code=401)

    transcript = payload.transcript

    if len(transcript) < 400:
        return {
            "analysis_status": "processed",
            "analysis_error": "",
            "analysis_last_run_at": now_iso(),
            "extraction_json": "{}",
        }

    try:

        enriched = enrich_transcript(transcript)

        prompt = build_prompt(
            payload.clari_call_id or "",
            enriched
        )

        result = call_claude(prompt)

        return {
            "analysis_status": "processed",
            "analysis_error": "",
            "analysis_last_run_at": now_iso(),
            "extraction_json": json.dumps(result),
        }

    except Exception as e:

        return {
            "analysis_status": "error",
            "analysis_error": str(e),
            "analysis_last_run_at": now_iso(),
            "extraction_json": "",
        }
