import re
from dataclasses import dataclass

_EMERGENCY_PATTERNS = [
    r"\bchest pain\b",
    r"\bshortness of breath\b",
    r"\bstroke\b",
    r"\bseizure\b",
    r"\bsuicid(al|e)\b",
    r"\boverdose\b",
    r"\bunconscious\b",
    r"\bsevere bleeding\b",
]

_DOSING_PATTERNS = [
    r"\bdos(e|age)\b",
    r"\bmg\b",
    r"\bml\b",
    r"\btablet(s)?\b",
    r"\bhow much\b",
]

@dataclass(frozen=True)
class SafetyResult:
    allowed: bool
    reason: str
    user_message: str

def check_safety(query: str) -> SafetyResult:
    q = query.lower()

    if any(re.search(p, q) for p in _EMERGENCY_PATTERNS):
        return SafetyResult(
            allowed=False,
            reason="emergency_risk",
            user_message=(
                "This may be an emergency. I can’t help with urgent medical decisions. "
                "Please contact local emergency services or a licensed clinician immediately."
            ),
        )

    if any(re.search(p, q) for p in _DOSING_PATTERNS):
        return SafetyResult(
            allowed=True,
            reason="dosing_caution",
            user_message=(
                "Caution: dosing and medication decisions should be confirmed with a licensed clinician. "
                "I will provide evidence-grounded information only from the supplied guidelines."
            ),
        )

    return SafetyResult(allowed=True, reason="ok", user_message="")