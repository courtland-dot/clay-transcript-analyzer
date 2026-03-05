import os
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Any

import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from requests.exceptions import ReadTimeout, ConnectTimeout

# ----------------------------
# Env
# ----------------------------
CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# IMPORTANT: Clay + Anthropic often needs > 18s
CLAUDE_TIMEOUT_SECS = int(os.environ.get("CLAUDE_TIMEOUT_SECS", "90"))
CLAUDE_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "3000"))

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


def clamp_str(s: str, n: int = 4000) -> str:
    return (s or "")[:n]


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
    Extract the first top-level JSON object from a string, even if there’s extra text.
    """
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty model output")

    # Fast path
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


def try_parse_json_loose(raw_text: str) -> dict:
    """
    Robust parser:
    1) extract_first_json_object
    2) json.loads
    3) if top-level is {'questions': '{...json...}'}, unwrap that
    """
    obj_text = extract_first_json_object(raw_text)
    parsed = json.loads(obj_text)

    # Handle annoying case: {"questions":"{ ... full schema ... }"} (stringified JSON)
    if isinstance(parsed, dict) and len(parsed) == 1:
        (k, v), = parsed.items()
        if isinstance(v, str):
            v_str = v.strip()
            if v_str.startswith("{") and v_str.endswith("}"):
                try:
                    inner = json.loads(v_str)
                    if isinstance(inner, dict) and "call_id" in inner:
                        return inner
                except Exception:
                    pass

    return parsed


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

    # display normalization
    if looks_like_email(display):
        speaker_display = display_from_email(display) or display
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
    out_lines: List[str] = []
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
def build_prompt(payload: AnalyzeIn, enriched: str, extra_strict: bool = False) -> str:
    strict_line = ""
    if extra_strict:
        strict_line = (
            "\nABSOLUTE REQUIREMENT: Return a single JSON object and NOTHING ELSE. "
            "Do not wrap JSON in quotes. Do not nest it inside another key. Do not add any text.\n"
        )

    return f"""
You are an information extraction engine. You must output VALID JSON ONLY (no markdown, no commentary).
Only extract items from lines labeled SPEAKER[prospect]. Ignore SPEAKER[regal] lines.
{strict_line}

CRITICAL CREDIBILITY RULES:
- Every extracted item MUST include an evidence_quote that is an exact, verbatim substring from the transcript provided.
- The evidence_quote MUST begin with "SPEAKER[prospect]" and MUST be taken from exactly ONE prospect line.
- evidence_quote MUST be SHORT: <= 320 characters. Use the smallest substring that still supports the item.
- If you cannot find a verbatim quote supporting an item, omit that item. Do not infer. Do not guess.
- Do not fabricate names, integrations, compliance requirements, numbers, timelines, outcomes, or next steps.
- Return empty arrays when nothing is found.

OUTPUT SIZE LIMITS (IMPORTANT):
- Max 8 questions, 8 objections, 8 product_feedback, 8 buying_signals.
- Keep "verbatim" <= 200 chars.
- Keep "normalized" <= 120 chars.
- Keep summary_10_lines exactly 10 short lines (<= 140 chars each).

Return JSON with this exact schema (top-level object):

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
- topic_tags MUST be grounded in the prospect transcript content.
- tags should include specific systems/terms if present in prospect words (salesforce, hubspot, whatsapp, meta, soc2, hipaa, tcpa, data_residency).

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
def call_claude_raw(prompt: str, max_tokens: int) -> str:
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

    # Use json= for correct encoding + headers still fine
    r = requests.post(url, headers=headers, json=body, timeout=CLAUDE_TIMEOUT_SECS)

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Anthropic error {r.status_code}: {err}")

    data = r.json()
    content = data.get("content", [])
    parts = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                parts.append(block["text"])

    raw_text = "\n".join(parts).strip()
    if not raw_text:
        raise ValueError("Claude response missing text content")

    return raw_text


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso(), "model": CLAUDE_MODEL, "timeout": CLAUDE_TIMEOUT_SECS}


@app.post("/analyze")
def analyze(payload: AnalyzeIn, authorization: str | None = Header(default=None)):
    if AUTH_TOKEN:
        if not authorization or authorization != f"Bearer {AUTH_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    call_id = payload.clari_call_id or ""
    transcript = payload.transcript or ""

    # --- Fast path: short transcript ---
    if len(transcript) < 400:
        base = {
            "call_id": call_id,
            "questions": [],
            "objections": [],
            "product_feedback": [],
            "buying_signals": [],
            "summary_10_lines": [],
            "topic_tags": [],
            "quality_flags": {"transcript_too_short": True, "low_signal": True},
        }
        return {
            "call_id": call_id,  # ✅ top-level join key for Clay
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
            "top_questions_bullets": "",
            "evidence_quotes_json": json.dumps([]),
            "topic_tag_evidence_json": json.dumps([]),
            "debug_model_output": "",
        }

    enriched = enrich_transcript_for_attribution(transcript)

    raw_text_first_attempt = ""
    raw_text_second_attempt = ""

    try:
        # ---- Attempt 1 ----
        prompt = build_prompt(payload, enriched, extra_strict=False)
        raw_text_first_attempt = call_claude_raw(prompt, CLAUDE_MAX_TOKENS)
        result = try_parse_json_loose(raw_text_first_attempt)

        if not isinstance(result, dict):
            raise ValueError("Model output parsed, but was not a JSON object")

    except (ValueError, json.JSONDecodeError):
        # ---- Attempt 2 (retry stricter + fewer tokens to reduce truncation) ----
        try:
            prompt2 = build_prompt(payload, enriched, extra_strict=True)
            raw_text_second_attempt = call_claude_raw(prompt2, min(CLAUDE_MAX_TOKENS, 1800))
            result = try_parse_json_loose(raw_text_second_attempt)

            if not isinstance(result, dict):
                raise ValueError("Model output parsed, but was not a JSON object")

        except Exception as e2:
            # Prefer the second attempt output if present, otherwise fall back to first attempt
            debug = raw_text_second_attempt or raw_text_first_attempt
            return {
                "call_id": call_id,
                "analysis_status": "error",
                "analysis_error": str(e2)[:4000],
                "analysis_last_run_at": now_iso(),
                "extraction_json": "",
                "questions_json": json.dumps([]),
                "objections_json": json.dumps([]),
                "product_feedback_json": json.dumps([]),
                "buying_signals_json": json.dumps([]),
                "summary_lines_json": json.dumps([]),
                "topic_tags": [],
                "top_questions_bullets": "",
                "evidence_quotes_json": json.dumps([]),
                "topic_tag_evidence_json": json.dumps([]),
                "debug_model_output": clamp_str(debug, 4000),  # ✅ always non-empty when Claude returned something
            }

    except (ReadTimeout, ConnectTimeout) as e:
        return {
            "call_id": call_id,
            "analysis_status": "error",
            "analysis_error": f"Anthropic timeout: {str(e)}"[:4000],
            "analysis_last_run_at": now_iso(),
            "extraction_json": "",
            "questions_json": json.dumps([]),
            "objections_json": json.dumps([]),
            "product_feedback_json": json.dumps([]),
            "buying_signals_json": json.dumps([]),
            "summary_lines_json": json.dumps([]),
            "topic_tags": [],
            "top_questions_bullets": "",
            "evidence_quotes_json": json.dumps([]),
            "topic_tag_evidence_json": json.dumps([]),
            "debug_model_output": clamp_str(raw_text_first_attempt, 4000),
        }

    except Exception as e:
        debug = raw_text_second_attempt or raw_text_first_attempt
        return {
            "call_id": call_id,
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
            "top_questions_bullets": "",
            "evidence_quotes_json": json.dumps([]),
            "topic_tag_evidence_json": json.dumps([]),
            "debug_model_output": clamp_str(debug, 4000),
        }

    # ----------------------------
    # Normalize / safe defaults
    # ----------------------------
    result["call_id"] = result.get("call_id") or call_id  # ✅ enforce call_id always present

    questions = result.get("questions", []) or []
    objections = result.get("objections", []) or []
    product_feedback = result.get("product_feedback", []) or []
    buying_signals = result.get("buying_signals", []) or []
    topic_tags = result.get("topic_tags", []) or []
    summary_lines = result.get("summary_10_lines", []) or []

    # Evidence quotes (dedupe)
    evidence_quotes: List[str] = []
    for arr in (questions, objections, product_feedback, buying_signals):
        for item in arr[:50]:
            if isinstance(item, dict) and item.get("evidence_quote"):
                evidence_quotes.append(item["evidence_quote"])

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

    return {
        "call_id": result.get("call_id", call_id),  # ✅ top-level join key
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
        "top_questions_bullets": "\n".join(top_bullets),
        "evidence_quotes_json": json.dumps(deduped_quotes[:25]),
        "topic_tag_evidence_json": json.dumps([]),
        "debug_model_output": "",
    }
