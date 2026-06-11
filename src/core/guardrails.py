"""
Guardrails layer for Smart Banking Assistant RAG API.

Validators:
 1. PII redaction    — GuardrailsPII applied to ANSWER
 2. Toxicity checker — ToxicLanguage applied to QUERY
"""

import os
import re
import uuid
from dotenv import load_dotenv

load_dotenv(override=True)

# Safe import across versions
try:
    from guardrails.errors import ValidationError
except Exception:
    ValidationError = Exception

PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "PERSON",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "IP_ADDRESS",
]

TOXICITY_THRESHOLD = float(os.getenv("GUARDRAIL_TOXICITY_THRESHOLD", "0.5"))
CUSTOMER_ID_RE = re.compile(r"\b\d{6,}\b")  # Mask numeric customer IDs

# GuardrailsPII (and our own substitution) emit <ENTITY_TYPE> placeholders.
# Streamlit's markdown renderer silently drops angle-bracket tags, leaving
# bare text like "account: " instead of "account: [ACCOUNT_NUMBER]".
# This regex rewrites all such tags to square-bracket form so they render.
_HTML_PII_TAG_RE = re.compile(r"<([A-Z][A-Z0-9_]*)>")

class GuardrailViolation(Exception):
    """Raised when input guardrail blocks a request."""
    def __init__(self, guard: str, message: str):
        self.guard = guard
        self.message = message
        super().__init__(f"[{guard}] {message}")

# ── Lazy guard construction
_guards = None

def _ensure_guardrails_configured() -> None:
    api_key = os.getenv("GUARDRAILS_API_KEY")
    if not api_key:
        return
    os.environ.setdefault("GUARDRAILS_TOKEN", api_key)
    rc_path = os.path.expanduser("~/.guardrailsrc")
    if os.path.exists(rc_path):
        return
    use_remote = os.getenv("GUARDRAILS_USE_REMOTE_INFERENCING", "false")
    try:
        with open(rc_path, "w") as rc_file:
            rc_file.write(
                f"id={uuid.uuid4()}\n"
                f"token={api_key}\n"
                "enable_metrics=false\n"
                f"use_remote_inferencing={use_remote}\n"
            )
    except OSError:
        pass

def _build_guards() -> dict:
    _ensure_guardrails_configured()
    try:
        from guardrails import Guard
        from guardrails.hub import GuardrailsPII, ToxicLanguage
    except ImportError as exc:
        raise RuntimeError(
            "Guardrails validators not installed. Run:\n"
            "  pip install guardrails-ai\n"
            "  guardrails configure\n"
            "  guardrails hub install hub://guardrails/guardrails_pii\n"
            "  guardrails hub install hub://guardrails/toxic_language"
        ) from exc

    return {
        "pii": Guard().use(GuardrailsPII(entities=PII_ENTITIES, on_fail="fix")),
        "toxicity": Guard().use(
            ToxicLanguage(
                threshold=TOXICITY_THRESHOLD,
                validation_method="sentence",
                on_fail="exception",
            )
        ),
    }

def _get_guards() -> dict:
    global _guards
    if _guards is None:
        _guards = _build_guards()
    return _guards

# ── Public API

def guard_input(query: str) -> None:
    """Raise GuardrailViolation if query is toxic."""
    guards = _get_guards()
    try:
        guards["toxicity"].validate(query)
    except ValidationError as exc:
        raise GuardrailViolation(
            "toxic_language",
            "Your message was flagged as abusive or toxic and cannot be processed.",
        ) from exc

def guard_output(answer: str) -> str:
    """Redact PII from the model's answer."""
    if not answer:
        return answer
    answer = CUSTOMER_ID_RE.sub("[CUSTOMER_ID]", answer)
    guards = _get_guards()
    outcome = guards["pii"].validate(answer)
    redacted = getattr(outcome, "validated_output", None) or answer
    # Rewrite any remaining <TAG> placeholders from GuardrailsPII to [TAG]
    # so Streamlit's markdown renderer displays them instead of dropping them.
    redacted = _HTML_PII_TAG_RE.sub(r"[\1]", redacted)
    return redacted