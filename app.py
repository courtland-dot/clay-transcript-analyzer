import os
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

# ----------------------------
# Env
# ----------------------------
CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]

# Default to a model that your /v1/models output shows exists
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
# e.g. "Alex Catalisan|Maaria Khalid|Courtland Nicholas"
EMPLOYEE_NAMES: List[str] = [
    n.strip() for n in os.environ.get("REGAL_EMPLOYEE_NAMES", "").split("|") if n.strip()
]
EMPLOYEE_NAMES_LOWER = {n.lower() for n in EMPLOYEE_NAMES}

# Tuning
CLAUDE_TIMEOUT_SECS = int(os.environ.get("CLAUDE_TIMEOUT_SECS", "120"))
MAX_WORKERS = int(os.environ.get("CLAUDE_PARALLEL_WORKERS", "5"))

# Keep each sub-call bounded so we don't blow Clay timeouts
MAX_TOKENS_QUESTIONS = int(os.environ.get("CLAUDE_MAX_TOKENS_QUESTIONS", "1200"))
MAX_TOKENS_OBJECTIONS = int(os.environ.get("CLAUDE_MAX_TOKENS_OBJECTIONS", "1200"))
MAX_TOKENS_FEEDBACK = int(os.environ.get("CLAUDE_MAX_TOKENS_FEEDBACK", "1200"))
MAX_TOKENS_SIGNALS = int(os.environ.get("CLAUDE_MAX_TOKENS_SIGNALS", "1200"))
MAX_TOKENS_SUMMARY = int(os.environ.get("CLAUDE_MAX_TOKENS_SUMMARY", "1600"))

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


def safe_list(x):
    return x if isinstance(x, list) else []


def safe_str(x):
    return x if isinstance(x, str) else ""


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
    """
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
    """
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
    Rewrite each speaker line:
      SPEAKER[prospect|regal] display=<...> email=<...> company=<...>: utterance...

    Enhancements:
    - Speaker identity memory within a single transcript:
        If we see "Patrice" with an email once, later "Patrice:" lines inherit that email/company.
    - Keeps original utterance unchanged.
    - Non-speaker lines pass through unchanged.
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


def filter_prospect_only(enriched: str) -> str:
    """
    Major speed win: send Claude only prospect speaker lines.
    Keeps evidence-quote rule intact (still exact substring of provided transcript).
    """
    lines = []
    for ln in (enriched or "").splitlines():
        if ln.startswith("SPEAKER[prospect]"):
            lines.append(ln)
    return "\n".join(lines)


# ----------------------------
# Prompting (split into 5 bounded calls)
# ----------------------------
def _prompt_header(payload: AnalyzeIn) -> str:
    call_id = payload.clari_call_id or ""
    sf_opp_id = payload.salesforce_opp_id or ""
    stage = payload.stage_at_time or ""
    segment = payload.segment or ""
    return f"""
You are an information extraction engine. You must output VALID JSON ONLY (no markdown, no backticks, no commentary).
Only extract items from lines labeled SPEAKER[prospect]. Ignore SPEAKER[regal] lines.

Every extracted item MUST include an evidence_quote that is an exact, verbatim substring from the transcript provided.
The evidence_quote MUST include the SPEAKER[prospect] prefix content verbatim as it appears.
If you cannot find a verbatim quote supporting an item, omit that item. Do not infer. Do not guess.
Do not fabricate names, integrations, compliance requirements, numbers, timelines, outcomes, or next steps.
Return empty arrays when nothing is found.

Inputs:
clari_call_id: {call_id}
salesforce_opp_id: {sf_opp_id}
stage_at_time: {stage}
segment: {segment}
""".strip()


def prompt_questions(payload: AnalyzeIn, transcript: str) -> str:
    call_id = payload.clari_call_id or ""
    return f"""{_prompt_header(payload)}

Extract buyer QUESTIONS only. Return JSON object with:
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
  ]
}}

Rules:
- Output JSON only.
- Keep verbatim <= 240 chars.
- normalized is a reusable FAQ/blog title.
- tags are specific terms present in prospect words (salesforce, hubspot, five9, soc2, hipaa, tcpa, etc.).
- confidence 0.0–1.0, conservative.

transcript:
{transcript}
""".strip()


def prompt_objections(payload: AnalyzeIn, transcript: str) -> str:
    call_id = payload.clari_call_id or ""
    return f"""{_prompt_header(payload)}

Extract buyer OBJECTIONS / concerns only. Return JSON object with:
{{
  "call_id": "{call_id}",
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
  ]
}}

Rules:
- Output JSON only.
- Keep verbatim <= 240 chars.
- confidence 0.0–1.0, conservative.

transcript:
{transcript}
""".strip()


def prompt_product_feedback(payload: AnalyzeIn, transcript: str) -> str:
    call_id = payload.clari_call_id or ""
    return f"""{_prompt_header(payload)}

Extract PRODUCT FEEDBACK moments only (requests, confusion, missing capability, competitor comparisons, implementation friction).
Return JSON object with:
{{
  "call_id": "{call_id}",
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
  ]
}}

Rules:
- Output JSON only.
- Keep verbatim <= 240 chars.

transcript:
{transcript}
""".strip()


def prompt_buying_signals(payload: AnalyzeIn, transcript: str) -> str:
    call_id = payload.clari_call_id or ""
    return f"""{_prompt_header(payload)}

Extract BUYING SIGNALS only. Return JSON object with:
{{
  "call_id": "{call_id}",
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
  ]
}}

Rules:
- Output JSON only.
- Keep verbatim <= 240 chars.
- confidence 0.0–1.0, conservative.

transcript:
{transcript}
""".strip()


def prompt_summary_and_tags(payload: AnalyzeIn, transcript: str) -> str:
    call_id = payload.clari_call_id or ""
    return f"""{_prompt_header(payload)}

Produce:
- summary_10_lines: 10 bullet-like lines (strings) grounded in the transcript.
- topic_tags: short snake_case tags grounded in transcript.
- topic_tag_evidence: mapping of topic_tag -> exact evidence_quote substring.

Return JSON object with:
{{
  "call_id": "{call_id}",
  "summary_10_lines": ["..."],
  "topic_tags": ["..."],
  "topic_tag_evidence": [
    {{
      "tag": "",
      "evidence_quote": ""
    }}
  ]
}}

Rules:
- Output JSON only.
- Evidence quotes MUST be verbatim substrings from transcript (include SPEAKER[prospect] prefix).
- Do not invent facts not in transcript.

transcript:
{transcript}
""".strip()


# ----------------------------
# Anthropic call
# ----------------------------
def call_claude_json(prompt: str, max_tokens: int) -> Tuple[dict, str]:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }

    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=CLAUDE_TIMEOUT_SECS)

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Anthropic error {r.status_code}: {err}")

    data = r.json()
    content = data.get("content", [])

    raw_text = ""
    if isinstance(content, list) and content:
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
            "summary_lines_json": json.dumps([]),
            "topic_tags": [],
            "topic_tag_evidence_json": json.dumps([]),
            "top_questions_bullets": "",
            "evidence_quotes_json": json.dumps([]),
        }

    # Enrich + strip to prospect-only for speed + stronger grounding
    enriched = enrich_transcript_for_attribution(transcript)
    prospect_only = filter_prospect_only(enriched)

    # If for some reason we lost everything (e.g., parsing mismatch), fall back to enriched
    transcript_for_model = prospect_only if prospect_only.strip() else enriched

    debug_raw_by_section: Dict[str, str] = {}

    def run_section(name: str, prompt: str, max_tokens: int):
        data, raw = call_claude_json(prompt, max_tokens=max_tokens)
        return name, data, raw

    try:
        prompts = [
            ("questions", prompt_questions(payload, transcript_for_model), MAX_TOKENS_QUESTIONS),
            ("objections", prompt_objections(payload, transcript_for_model), MAX_TOKENS_OBJECTIONS),
            ("product_feedback", prompt_product_feedback(payload, transcript_for_model), MAX_TOKENS_FEEDBACK),
            ("buying_signals", prompt_buying_signals(payload, transcript_for_model), MAX_TOKENS_SIGNALS),
            ("summary_pack", prompt_summary_and_tags(payload, transcript_for_model), MAX_TOKENS_SUMMARY),
        ]

        results: Dict[str, dict] = {}

        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(prompts))) as ex:
            futures = [ex.submit(run_section, n, p, mt) for (n, p, mt) in prompts]
            for fut in as_completed(futures):
                name, data, raw = fut.result()
                results[name] = data
                debug_raw_by_section[name] = raw[:2000]

        # Pull out sections safely
        questions = safe_list(results.get("questions", {}).get("questions"))
        objections = safe_list(results.get("objections", {}).get("objections"))
        product_feedback = safe_list(results.get("product_feedback", {}).get("product_feedback"))
        buying_signals = safe_list(results.get("buying_signals", {}).get("buying_signals"))

        summary_pack = results.get("summary_pack", {}) or {}
        summary_10_lines = safe_list(summary_pack.get("summary_10_lines"))
        topic_tags = safe_list(summary_pack.get("topic_tags"))
        topic_tag_evidence = safe_list(summary_pack.get("topic_tag_evidence"))

        # Evidence quotes for convenience columns
        evidence_quotes = []
        for q in questions[:25]:
            if isinstance(q, dict) and q.get("evidence_quote"):
                evidence_quotes.append(q["evidence_quote"])
        for o in objections[:25]:
            if isinstance(o, dict) and o.get("evidence_quote"):
                evidence_quotes.append(o["evidence_quote"])

        seen = set()
        deduped_quotes = []
        for q in evidence_quotes:
            if q not in seen:
                seen.add(q)
                deduped_quotes.append(q)

        top_bullets = []
        for q in questions[:7]:
            if isinstance(q, dict):
                norm = q.get("normalized") or q.get("verbatim")
                if norm:
                    top_bullets.append(f"- {norm}")

        merged = {
            "call_id": payload.clari_call_id or "",
            "questions": questions,
            "objections": objections,
            "product_feedback": product_feedback,
            "buying_signals": buying_signals,
            "summary_10_lines": summary_10_lines,
            "topic_tags": topic_tags,
            "quality_flags": {"transcript_too_short": False, "low_signal": False},
        }

        return {
            "analysis_status": "processed",
            "analysis_error": "",
            "analysis_last_run_at": now_iso(),
            "extraction_json": json.dumps(merged),
            "questions_json": json.dumps(questions),
            "objections_json": json.dumps(objections),
            "product_feedback_json": json.dumps(product_feedback),
            "buying_signals_json": json.dumps(buying_signals),
            "summary_lines_json": json.dumps(summary_10_lines),
            "topic_tags": topic_tags,
            "topic_tag_evidence_json": json.dumps(topic_tag_evidence),
            "top_questions_bullets": "\n".join(top_bullets),
            "evidence_quotes_json": json.dumps(deduped_quotes[:25]),
        }

    except Exception as e:
        # Return error fields Clay can map + include per-section debug
        return {
            "analysis_status": "error",
            "analysis_error": str(e)[:4000],
            "analysis_last_run_at": now_iso(),
            "extraction_json": "",
            "questions_json": json.dumps([]),
            "objections_json": json.dumps([]),
            "product_feedback_json": json.dumps([]),
            "buying_signals_json": json.dumps([]),
            "summary_lines_json": json.dumps([]),
            "topic_tags": [],
            "topic_tag_evidence_json": json.dumps([]),
            "top_questions_bullets": "",
            "evidence_quotes_json": json.dumps([]),
            "debug_model_output": json.dumps(debug_raw_by_section)[:4000],
        }
