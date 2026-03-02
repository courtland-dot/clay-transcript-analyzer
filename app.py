import os
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Any

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

# Tuneable knobs
CLAUDE_TIMEOUT_SEC = int(os.environ.get("CLAUDE_TIMEOUT_SEC", "120"))
CLAUDE_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "4000"))  # per-call
# If you still see truncation, raise to 6000; multi-call should prevent it anyway.

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

def extract_first_json_value(text: str) -> str:
    """
    Extract first top-level JSON value (object or array) from a string.
    Handles extra text around it. Raises ValueError if incomplete/truncated.
    """
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty model output")

    # fast paths
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        return s

    # find first { or [
    start_obj = s.find("{")
    start_arr = s.find("[")
    if start_obj == -1 and start_arr == -1:
        raise ValueError("No JSON start '{' or '[' found in model output")

    if start_obj == -1:
        start = start_arr
        open_ch, close_ch = "[", "]"
    elif start_arr == -1:
        start = start_obj
        open_ch, close_ch = "{", "}"
    else:
        # earliest
        if start_obj < start_arr:
            start = start_obj
            open_ch, close_ch = "{", "}"
        else:
            start = start_arr
            open_ch, close_ch = "[", "]"

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
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]

            # allow nested other bracket types without counting them
            # (JSON guarantees balanced overall; we only track the first type)

    raise ValueError("Could not find a complete JSON value in model output (likely truncated)")

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

    # Display label
    if looks_like_email(display):
        pretty = display_from_email(display)
        speaker_display = pretty or display
    else:
        speaker_display = display or (display_from_email(email) if email else "Unknown")

    # suppress internal email
    speaker_email = "" if regal else (email or "")

    # prospect company derived from email domain if known
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
    Rewrite each speaker line to:
      SPEAKER[prospect|regal] display=<...> email=<...> company=<...>: utterance...

    Adds in-transcript identity memory: if we see a name with an email once,
    later lines with just the name inherit that email/company.
    """
    out_lines: List[str] = []
    speaker_memory: Dict[str, Tuple[str, str]] = {}  # display(lower) -> (email, dom)

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
# Anthropic call (raw)
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
        "system": "Return ONLY valid JSON. No extra text.",
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=CLAUDE_TIMEOUT_SEC)

    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Anthropic error {r.status_code}: {err}")

    data = r.json()
    content = data.get("content", [])

    parts: List[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                parts.append(block["text"])

    return "\n".join(parts).strip()

def call_claude_json(prompt: str, max_tokens: int) -> Tuple[Any, str]:
    raw = call_claude_raw(prompt, max_tokens=max_tokens)
    json_text = extract_first_json_value(raw)
    return json.loads(json_text), raw

# ----------------------------
# Prompts (small, per-section)
# ----------------------------
def common_instructions(enriched_transcript: str) -> str:
    return f"""
Only extract from lines starting with SPEAKER[prospect]. Ignore SPEAKER[regal].

Evidence rules:
- Every item MUST include evidence_quote which is an exact, verbatim substring of the transcript.
- evidence_quote MUST include the SPEAKER[prospect] prefix exactly as shown.
- If you cannot support it with an exact quote, OMIT it (do not guess).

Return JSON only. No markdown. No commentary.

Transcript:
{enriched_transcript}
""".strip()

def prompt_questions(enriched: str, limit: int = 10) -> str:
    return f"""
{common_instructions(enriched)}

Return a JSON array of questions (max {limit}).
Each item schema:
{{
  "verbatim": "",
  "speaker_display": "",
  "speaker_email": "",
  "speaker_company": "",
  "speaker_type": "prospect",
  "normalized": "",
  "category": "integration|security|pricing|implementation|product|roi|timeline|other",
  "tags": [],
  "evidence_quote": "",
  "confidence": 0.0
}}

Constraints:
- Keep verbatim <= 240 chars.
- Keep evidence_quote <= 260 chars (truncate the quote if longer but keep it verbatim substring).
""".strip()

def prompt_objections(enriched: str, limit: int = 12) -> str:
    return f"""
{common_instructions(enriched)}

Return a JSON array of objections/concerns (max {limit}).
Each item schema:
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

Constraints:
- Keep verbatim <= 240 chars.
- Keep evidence_quote <= 260 chars.
""".strip()

def prompt_product_feedback(enriched: str, limit: int = 10) -> str:
    return f"""
{common_instructions(enriched)}

Return a JSON array of product feedback moments (max {limit}).
Each item schema:
{{
  "type": "feature_request|bug|confusion|missing_capability|competitor_comparison|implementation_friction",
  "verbatim": "",
  "speaker_display": "",
  "speaker_email": "",
  "speaker_company": "",
  "speaker_type": "prospect",
  "evidence_quote": ""
}}

Constraints:
- Keep verbatim <= 240 chars.
- Keep evidence_quote <= 260 chars.
""".strip()

def prompt_buying_signals(enriched: str, limit: int = 10) -> str:
    return f"""
{common_instructions(enriched)}

Return a JSON array of buying signals (max {limit}).
Each item schema:
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

Constraints:
- Keep verbatim <= 240 chars.
- Keep evidence_quote <= 260 chars.
""".strip()

def prompt_summary_and_tags(enriched: str) -> str:
    return f"""
{common_instructions(enriched)}

Return ONE JSON object with:
{{
  "summary_10_lines": [{{"line":"","evidence_quote":""}}],
  "topic_tags": [],
  "topic_tag_evidence": [{{"topic_tag":"","evidence_quote":""}}],
  "quality_flags": {{"transcript_too_short": false, "low_signal": false}}
}}

Rules:
- summary_10_lines: max 10 items; each MUST have evidence_quote (<= 260 chars).
- topic_tags: max 15; lowercase snake_case; must be grounded in transcript.
- topic_tag_evidence: max 25; each evidence_quote <= 260 chars.
""".strip()

# ----------------------------
# Helpers for response shaping
# ----------------------------
def safe_list(x) -> list:
    return x if isinstance(x, list) else []

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in items:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

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

    debug_blobs: List[str] = []
    try:
        enriched = enrich_transcript_for_attribution(transcript)

        # 1) Questions
        questions, raw_q = call_claude_json(prompt_questions(enriched), max_tokens=CLAUDE_MAX_TOKENS)
        debug_blobs.append(("questions:\n" + raw_q)[:1200])

        # 2) Objections
        objections, raw_o = call_claude_json(prompt_objections(enriched), max_tokens=CLAUDE_MAX_TOKENS)
        debug_blobs.append(("objections:\n" + raw_o)[:1200])

        # 3) Product feedback
        product_feedback, raw_pf = call_claude_json(prompt_product_feedback(enriched), max_tokens=CLAUDE_MAX_TOKENS)
        debug_blobs.append(("product_feedback:\n" + raw_pf)[:1200])

        # 4) Buying signals
        buying_signals, raw_bs = call_claude_json(prompt_buying_signals(enriched), max_tokens=CLAUDE_MAX_TOKENS)
        debug_blobs.append(("buying_signals:\n" + raw_bs)[:1200])

        # 5) Summary + tags
        summary_pack, raw_sum = call_claude_json(prompt_summary_and_tags(enriched), max_tokens=CLAUDE_MAX_TOKENS)
        debug_blobs.append(("summary_and_tags:\n" + raw_sum)[:1200])

        questions = safe_list(questions)
        objections = safe_list(objections)
        product_feedback = safe_list(product_feedback)
        buying_signals = safe_list(buying_signals)

        summary_lines_json: List[str] = []
        summary_evidence: List[str] = []
        topic_tags: List[str] = []
        topic_tag_evidence_quotes: List[str] = []
        quality_flags = {"transcript_too_short": False, "low_signal": False}

        if isinstance(summary_pack, dict):
            # summary lines as list of dicts
            for item in safe_list(summary_pack.get("summary_10_lines", []))[:10]:
                if isinstance(item, dict):
                    line = (item.get("line") or "").strip()
                    ev = (item.get("evidence_quote") or "").strip()
                    if line:
                        summary_lines_json.append(line)
                    if ev:
                        summary_evidence.append(ev)

            topic_tags = [t for t in safe_list(summary_pack.get("topic_tags", [])) if isinstance(t, str)][:15]

            for item in safe_list(summary_pack.get("topic_tag_evidence", []))[:25]:
                if isinstance(item, dict):
                    ev = (item.get("evidence_quote") or "").strip()
                    if ev:
                        topic_tag_evidence_quotes.append(ev)

            qf = summary_pack.get("quality_flags")
            if isinstance(qf, dict):
                quality_flags = {
                    "transcript_too_short": bool(qf.get("transcript_too_short", False)),
                    "low_signal": bool(qf.get("low_signal", False)),
                }

        # Evidence quotes aggregate (questions + objections + summary + topic_tag_evidence)
        evidence_quotes: List[str] = []
        for q in questions[:25]:
            if isinstance(q, dict) and q.get("evidence_quote"):
                evidence_quotes.append(q["evidence_quote"])
        for o in objections[:25]:
            if isinstance(o, dict) and o.get("evidence_quote"):
                evidence_quotes.append(o["evidence_quote"])
        evidence_quotes.extend(summary_evidence[:25])
        evidence_quotes.extend(topic_tag_evidence_quotes[:25])
        evidence_quotes = dedupe_preserve_order(evidence_quotes)[:25]

        # bullets
        top_bullets = []
        for q in questions[:7]:
            if isinstance(q, dict):
                norm = q.get("normalized") or q.get("verbatim")
                if norm:
                    top_bullets.append(f"- {norm}")

        result_obj = {
            "call_id": payload.clari_call_id or "",
            "questions": questions,
            "objections": objections,
            "product_feedback": product_feedback,
            "buying_signals": buying_signals,
            "summary_10_lines": summary_lines_json,
            "topic_tags": topic_tags,
            "topic_tag_evidence": topic_tag_evidence_quotes,
            "quality_flags": quality_flags,
        }

        return {
            "analysis_status": "processed",
            "analysis_error": "",
            "analysis_last_run_at": now_iso(),
            "extraction_json": json.dumps(result_obj),
            "questions_json": json.dumps(questions),
            "objections_json": json.dumps(objections),
            "product_feedback_json": json.dumps(product_feedback),
            "buying_signals_json": json.dumps(buying_signals),
            "summary_lines_json": json.dumps(summary_lines_json),
            "topic_tags": topic_tags,
            "topic_tag_evidence_json": json.dumps(topic_tag_evidence_quotes),
            "top_questions_bullets": "\n".join(top_bullets),
            "evidence_quotes_json": json.dumps(evidence_quotes),
            "debug_model_output": "\n\n---\n\n".join(debug_blobs)[:4000],
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
            "summary_lines_json": json.dumps([]),
            "topic_tags": [],
            "topic_tag_evidence_json": json.dumps([]),
            "top_questions_bullets": "",
            "evidence_quotes_json": json.dumps([]),
            "debug_model_output": ("\n\n---\n\n".join(debug_blobs)[:4000] if debug_blobs else ""),
        }
