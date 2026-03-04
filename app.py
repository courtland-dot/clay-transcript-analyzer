import os
import json
import re
import time
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

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
# e.g. "Alex Krumm|Maaria Khalid|Courtland Nicholas"
EMPLOYEE_NAMES: List[str] = [
    n.strip() for n in os.environ.get("REGAL_EMPLOYEE_NAMES", "").split("|") if n.strip()
]
EMPLOYEE_NAMES_LOWER = {n.lower() for n in EMPLOYEE_NAMES}

# Timeouts / performance knobs (set these in Render env)
CLAUDE_TIMEOUT_SECS = int(os.environ.get("CLAUDE_TIMEOUT_SECS", "18"))
ANALYZE_OVERALL_TIMEOUT_SECS = int(os.environ.get("ANALYZE_OVERALL_TIMEOUT_SECS", "25"))
CLAUDE_PARALLEL_WORKERS = int(os.environ.get("CLAUDE_PARALLEL_WORKERS", "3"))

# Per-section token budgets (set in Render env; defaults are safe)
CLAUDE_MAX_TOKENS_QUESTIONS = int(os.environ.get("CLAUDE_MAX_TOKENS_QUESTIONS", "900"))
CLAUDE_MAX_TOKENS_OBJECTIONS = int(os.environ.get("CLAUDE_MAX_TOKENS_OBJECTIONS", "900"))
CLAUDE_MAX_TOKENS_FEEDBACK = int(os.environ.get("CLAUDE_MAX_TOKENS_FEEDBACK", "900"))
CLAUDE_MAX_TOKENS_SIGNALS = int(os.environ.get("CLAUDE_MAX_TOKENS_SIGNALS", "900"))
CLAUDE_MAX_TOKENS_SUMMARY = int(os.environ.get("CLAUDE_MAX_TOKENS_SUMMARY", "900"))

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
    local = re.sub(r"[._\\-]+", " ", local).strip()
    return " ".join(w.capitalize() for w in local.split() if w)


def looks_like_email(s: str) -> bool:
    s = (s or "").strip()
    return bool(re.match(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", s))


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

    if email and domain_from_email(email) in REGAL_DOMAINS:
        return True

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

    # Prefer display name as given; if it's an email, prettify.
    if looks_like_email(display):
        pretty = display_from_email(display)
        speaker_display = pretty or display
    else:
        speaker_display = display or (display_from_email(email) if email else "Unknown")

    # Email/company only for prospects
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

        parts = line.split(":", 1)
        if len(parts) < 2:
            out_lines.append(raw_line)
            continue
        utterance = parts[1].strip()

        key = (display or "").strip().lower()

        # Remember email/domain when present
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
# Prompting (multi-call)
# ----------------------------
def _common_header(payload: AnalyzeIn, enriched_transcript: str) -> str:
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

Inputs:
clari_call_id: {call_id}
salesforce_opp_id: {sf_opp_id}
stage_at_time: {stage}
segment: {segment}

transcript:
{enriched_transcript}
""".strip()


def build_prompt_questions(payload: AnalyzeIn, enriched_transcript: str) -> str:
    call_id = payload.clari_call_id or ""
    return f"""
{_common_header(payload, enriched_transcript)}

Extract ONLY buyer questions asked by SPEAKER[prospect].

Return JSON with schema:
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
- normalized should be reusable as a FAQ/blog title.
- tags should include specific systems/terms if present (salesforce, hubspot, whatsapp, meta, soc2, hipaa, tcpa, data_residency).
""".strip()


def build_prompt_objections(payload: AnalyzeIn, enriched_transcript: str) -> str:
    call_id = payload.clari_call_id or ""
    return f"""
{_common_header(payload, enriched_transcript)}

Extract ONLY buyer objections / hesitations / concerns voiced by SPEAKER[prospect].

Return JSON with schema:
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
""".strip()


def build_prompt_feedback(payload: AnalyzeIn, enriched_transcript: str) -> str:
    call_id = payload.clari_call_id or ""
    return f"""
{_common_header(payload, enriched_transcript)}

Extract ONLY product feedback moments voiced by SPEAKER[prospect] (feature requests, missing capabilities, confusion, bugs, competitor comparisons, implementation friction).

Return JSON with schema:
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
""".strip()


def build_prompt_signals(payload: AnalyzeIn, enriched_transcript: str) -> str:
    call_id = payload.clari_call_id or ""
    return f"""
{_common_header(payload, enriched_transcript)}

Extract ONLY buying signals from SPEAKER[prospect].

Return JSON with schema:
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
""".strip()


def build_prompt_summary(payload: AnalyzeIn, enriched_transcript: str) -> str:
    call_id = payload.clari_call_id or ""
    return f"""
{_common_header(payload, enriched_transcript)}

Create:
1) summary_10_lines: exactly 10 short lines grounded in the transcript (no guessing).
2) topic_tags: 5-20 snake_case tags grounded in the transcript.
3) topic_tag_evidence: list of objects mapping each topic_tag -> one exact evidence_quote substring from transcript.

Return JSON with schema:
{{
  "call_id": "{call_id}",
  "summary_10_lines": ["..."],
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
- topic_tag_evidence.evidence_quote MUST be an exact substring (include SPEAKER[...] prefix if the line has it).
- quality_flags.transcript_too_short should be true if the transcript is short / low context.
- low_signal should be true if there are no clear questions/objections/feedback/signals.
""".strip()


# ----------------------------
# Anthropic call
# ----------------------------
def _maybe_json_load(s: str):
    """
    Try to json.loads a string if it looks like JSON; otherwise return None.
    """
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not t:
        return None
    if not (t.startswith("{") or t.startswith("[")):
        return None
    try:
        return json.loads(t)
    except Exception:
        return None


def _coerce_double_encoded(obj):
    """
    Fix common Claude failure mode: fields that should be lists/objects are returned
    as JSON-encoded strings.

    Examples:
      {"questions": "{...full schema...}"}   -> returns the inner object
      {"questions": "[{...}]", ...}          -> questions becomes a list
    """
    # If the entire object is a JSON string, decode it
    if isinstance(obj, str):
        decoded = _maybe_json_load(obj)
        return decoded if decoded is not None else obj

    if not isinstance(obj, dict):
        return obj

    # If it's a wrapper dict with a single key whose value is JSON, unwrap
    if len(obj) == 1:
        only_val = next(iter(obj.values()))
        if isinstance(only_val, str):
            decoded = _maybe_json_load(only_val)
            if isinstance(decoded, dict):
                return _coerce_double_encoded(decoded)

    # Otherwise, walk fields and decode any JSON-looking strings
    fixed = {}
    for k, v in obj.items():
        if isinstance(v, str):
            decoded = _maybe_json_load(v)
            fixed[k] = decoded if decoded is not None else v
        elif isinstance(v, dict):
            fixed[k] = _coerce_double_encoded(v)
        elif isinstance(v, list):
            # Also decode list elements that are JSON strings
            new_list = []
            for item in v:
                if isinstance(item, str):
                    decoded_item = _maybe_json_load(item)
                    new_list.append(decoded_item if decoded_item is not None else item)
                else:
                    new_list.append(_coerce_double_encoded(item) if isinstance(item, dict) else item)
            fixed[k] = new_list
        else:
            fixed[k] = v

    return fixed


def _parse_claude_json(raw_text: str) -> dict:
    """
    Parse Claude output into a dict, tolerating:
    - pre/post text around JSON
    - double-encoded JSON fields
    """
    t = (raw_text or "").strip()
    if not t:
        raise ValueError("Empty model output")

    # If the model (or proxy) returned HTML, fail fast with a clearer error
    if t.startswith("<!DOCTYPE") or t.startswith("<html") or "<title>502</title>" in t:
        raise ValueError("Upstream returned HTML (likely 502/timeout).")

    # Try direct JSON parse first
    direct = _maybe_json_load(t)
    if isinstance(direct, dict):
        return _coerce_double_encoded(direct)

    # Otherwise extract first object and parse
    json_text = extract_first_json_object(t)
    parsed = json.loads(json_text)

    if isinstance(parsed, dict):
        return _coerce_double_encoded(parsed)

    # If it's a list or something else, return as error (schema expects dict)
    raise ValueError("Model output JSON was not an object")


def call_claude(prompt: str, max_tokens: int) -> Tuple[dict, str]:
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

    r = requests.post(
        url,
        headers=headers,
        data=json.dumps(body),
        timeout=CLAUDE_TIMEOUT_SECS,
    )

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Anthropic error {r.status_code}: {err}")

    data = r.json()
    content = data.get("content", [])

    # Anthropic returns blocks like [{"type":"text","text":"..."}]
    raw_text_parts = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                raw_text_parts.append(block["text"])
    raw_text = "\n".join(raw_text_parts).strip()

    if not raw_text:
        # include the full response body for debugging if content is missing
        raise ValueError(f"Claude response missing text content. keys={list(data.keys())}")

    parsed = _parse_claude_json(raw_text)
    return parsed, raw_text

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "ts": now_iso(),
        "model": CLAUDE_MODEL,
        "claude_timeout_secs": CLAUDE_TIMEOUT_SECS,
        "analyze_overall_timeout_secs": ANALYZE_OVERALL_TIMEOUT_SECS,
        "parallel_workers": CLAUDE_PARALLEL_WORKERS,
    }


@app.post("/analyze")
def analyze(payload: AnalyzeIn, authorization: str | None = Header(default=None)):
    # Optional shared-secret gate
    if AUTH_TOKEN:
        if not authorization or authorization != f"Bearer {AUTH_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    transcript = payload.transcript or ""

    # Keep original early-return behavior for very short transcripts
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
            "summary_lines_json": json.dumps([]),
            "topic_tags": [],
            "topic_tag_evidence_json": json.dumps([]),
            "top_questions_bullets": "",
            "evidence_quotes_json": json.dumps([]),
            "debug_model_output": "",
        }

    enriched = enrich_transcript_for_attribution(transcript)

    # Build prompts for 5 calls
    prompts = [
        ("questions", build_prompt_questions(payload, enriched), CLAUDE_MAX_TOKENS_QUESTIONS),
        ("objections", build_prompt_objections(payload, enriched), CLAUDE_MAX_TOKENS_OBJECTIONS),
        ("product_feedback", build_prompt_feedback(payload, enriched), CLAUDE_MAX_TOKENS_FEEDBACK),
        ("buying_signals", build_prompt_signals(payload, enriched), CLAUDE_MAX_TOKENS_SIGNALS),
        ("summary", build_prompt_summary(payload, enriched), CLAUDE_MAX_TOKENS_SUMMARY),
    ]

    results: Dict[str, dict] = {}
    debug_raw_by_section: Dict[str, str] = {}

    def run_section(name: str, prompt: str, mt: int) -> Tuple[str, dict, str]:
        data, raw = call_claude(prompt, max_tokens=mt)
        return name, data, raw

    raw_text = ""
    try:
        # Hard overall deadline to avoid Render/Clay upstream timeouts (prevents 502 HTML)
        deadline = time.monotonic() + ANALYZE_OVERALL_TIMEOUT_SECS

        with ThreadPoolExecutor(max_workers=min(CLAUDE_PARALLEL_WORKERS, len(prompts))) as ex:
            future_map = {ex.submit(run_section, n, p, mt): n for (n, p, mt) in prompts}

            while future_map:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    for fut in future_map:
                        fut.cancel()
                    raise TimeoutError(
                        f"Analyze exceeded {ANALYZE_OVERALL_TIMEOUT_SECS}s overall timeout"
                    )

                done, _ = wait(
                    future_map.keys(),
                    timeout=min(remaining, 1.5),
                    return_when=FIRST_COMPLETED,
                )

                for fut in done:
                    name = future_map.pop(fut)
                    sec_name, data, raw = fut.result()
                    results[sec_name] = data
                    debug_raw_by_section[sec_name] = (raw or "")[:2000]

        # Merge results into single extraction schema
        questions = (results.get("questions", {}) or {}).get("questions", []) or []
        objections = (results.get("objections", {}) or {}).get("objections", []) or []
        product_feedback = (results.get("product_feedback", {}) or {}).get("product_feedback", []) or []
        buying_signals = (results.get("buying_signals", {}) or {}).get("buying_signals", []) or []

        summary_block = results.get("summary", {}) or {}
        summary_10_lines = summary_block.get("summary_10_lines", []) or []
        topic_tags = summary_block.get("topic_tags", []) or []
        topic_tag_evidence = summary_block.get("topic_tag_evidence", []) or []
        quality_flags = summary_block.get("quality_flags", {}) or {"transcript_too_short": False, "low_signal": False}

        # Evidence quotes (deduped)
        evidence_quotes = []
        for q in questions[:50]:
            eq = q.get("evidence_quote")
            if eq:
                evidence_quotes.append(eq)
        for o in objections[:50]:
            eq = o.get("evidence_quote")
            if eq:
                evidence_quotes.append(eq)
        for pf in product_feedback[:50]:
            eq = pf.get("evidence_quote")
            if eq:
                evidence_quotes.append(eq)
        for bs in buying_signals[:50]:
            eq = bs.get("evidence_quote")
            if eq:
                evidence_quotes.append(eq)
        for te in topic_tag_evidence[:50]:
            eq = (te or {}).get("evidence_quote")
            if eq:
                evidence_quotes.append(eq)

        seen = set()
        deduped_quotes = []
        for q in evidence_quotes:
            if q not in seen:
                seen.add(q)
                deduped_quotes.append(q)

        # Top questions bullets
        top_bullets = []
        for q in questions[:7]:
            norm = q.get("normalized") or q.get("verbatim")
            if norm:
                top_bullets.append(f"- {norm}")

        extraction = {
            "call_id": payload.clari_call_id or "",
            "questions": questions,
            "objections": objections,
            "product_feedback": product_feedback,
            "buying_signals": buying_signals,
            "summary_10_lines": summary_10_lines,
            "topic_tags": topic_tags,
            "topic_tag_evidence": topic_tag_evidence,
            "quality_flags": quality_flags,
        }

        # Provide compact debug output to Clay if needed
        debug_model_output = json.dumps(debug_raw_by_section)[:4000]

        return {
            "analysis_status": "processed",
            "analysis_error": "",
            "analysis_last_run_at": now_iso(),
            "extraction_json": json.dumps(extraction),
            "questions_json": json.dumps(questions),
            "objections_json": json.dumps(objections),
            "product_feedback_json": json.dumps(product_feedback),
            "buying_signals_json": json.dumps(buying_signals),
            "summary_lines_json": json.dumps(summary_10_lines),
            "topic_tags": topic_tags,
            "topic_tag_evidence_json": json.dumps(topic_tag_evidence),
            "top_questions_bullets": "\n".join(top_bullets),
            "evidence_quotes_json": json.dumps(deduped_quotes[:25]),
            "debug_model_output": debug_model_output,
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
            "summary_lines_json": json.dumps([]),
            "topic_tags": [],
            "topic_tag_evidence_json": json.dumps([]),
            "top_questions_bullets": "",
            "evidence_quotes_json": json.dumps([]),
            "debug_model_output": json.dumps(debug_raw_by_section)[:4000] if debug_raw_by_section else "",
        }
