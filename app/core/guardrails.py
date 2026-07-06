import os
import re

MAX_CONTEXT_STATE_LEN = int(os.getenv("GUARD_MAX_CONTEXT_LEN", "4000"))
MAX_ACTIONS = int(os.getenv("GUARD_MAX_ACTIONS", "20"))
MAX_ACTION_LEN = int(os.getenv("GUARD_MAX_ACTION_LEN", "500"))
AGENT_PYTHON_TOOL = os.getenv("AGENT_PYTHON_TOOL", "false").lower() in {"1", "true", "yes", "on"}


def validate_forecast(context_state: str, candidate_actions: list[str]) -> None:
    if not context_state.strip():
        raise ValueError("context_state required")
    if len(context_state) > MAX_CONTEXT_STATE_LEN:
        raise ValueError(f"context_state max {MAX_CONTEXT_STATE_LEN} chars")
    if not candidate_actions:
        raise ValueError("candidate_actions required")
    if len(candidate_actions) > MAX_ACTIONS:
        raise ValueError(f"max {MAX_ACTIONS} actions")
    for a in candidate_actions:
        if not a.strip():
            raise ValueError("empty action")
        if len(a) > MAX_ACTION_LEN:
            raise ValueError(f"action max {MAX_ACTION_LEN} chars")


def validate_url(url: str) -> None:
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must be http(s)")


def sanitize_text(text: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    return text[:50000]
