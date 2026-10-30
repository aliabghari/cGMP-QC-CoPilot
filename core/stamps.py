"""
cGMP-QC-CoPilot: Regulatory Stamping and Claims Linting Module.

Enforces Non-Negotiable Guardrails:
- G2: Non-GMP decision support. Every output stamped DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL.
      Banned phrases: "release for use", "zero risk", "validated", "guaranteed", "proves safe".

Supports negation and quotation awareness:
- Legitimate disclaimers (e.g. "not validated for GMP use", "cannot be guaranteed", "does not prove safe")
  and quoted references (e.g. 'validated', "zero risk") are permitted.
- Affirmative claims (e.g. "this model is validated", "we guarantee that", "release for use") are blocked.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

DRAFT_STAMP = "DRAFT - NOT A GMP RECORD - REQUIRES HUMAN QA APPROVAL"
PART11_DISCLAIMER = "Part 11-style demo only, not a compliant e-signature system."
SYNTHETIC_DATA_NOTICE = "SYNTHETIC / NOT GMP DATA — Prototype Decision-Support Modules for Demonstration Only"

# Exact list of banned phrases as mandated by Guardrail G2
BANNED_PHRASES: List[str] = [
    "release for use",
    "zero risk",
    "validated",
    "guaranteed",
    "proves safe",
    "GMP-ready",
    "audit-ready",
    "inspection-ready",
    "production-grade",
]

# Patterns supporting morphological variants (e.g. validate/validated, guarantee/guaranteed, prove safe/proves safe)
BANNED_PHRASE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("release for use", re.compile(r"\brelease(?:d)?\s+for\s+use\b", re.IGNORECASE)),
    ("zero risk", re.compile(r"\bzero\s+risk\b", re.IGNORECASE)),
    ("validated", re.compile(r"\bvalidate(?:d|s)?\b", re.IGNORECASE)),
    ("guaranteed", re.compile(r"\bguarantee(?:d|s)?\b", re.IGNORECASE)),
    ("proves safe", re.compile(r"\bprove(?:d|s)?\s+safe\b", re.IGNORECASE)),
    ("GMP-ready", re.compile(r"\bgmp[- ]ready\b", re.IGNORECASE)),
    ("audit-ready", re.compile(r"\baudit[- ]ready\b", re.IGNORECASE)),
    ("inspection-ready", re.compile(r"\binspection[- ]ready\b", re.IGNORECASE)),
    ("production-grade", re.compile(r"\bproduction[- ]grade\b", re.IGNORECASE)),
]

# Negation prefix tokens (case-insensitive)
NEGATION_WORDS = {
    "not",
    "no",
    "non",
    "never",
    "neither",
    "nor",
    "cannot",
    "cant",
    "can't",
    "without",
    "prohibit",
    "prohibits",
    "prohibited",
    "banned",
    "unvalidated",
    "reject",
    "disclaimer",
}


def _is_negated_or_quoted(text: str, start: int, end: int) -> bool:
    """
    Checks whether a match spanning text[start:end] is negated or inside quotes/backticks.
    """
    # 1. Check surrounding quotes or backticks (immediate or adjacent)
    char_before = text[start - 1] if start > 0 else ""
    char_after = text[end] if end < len(text) else ""
    if (char_before in ('"', "'", "`") and char_after in ('"', "'", "`")) or \
       (start >= 2 and text[start - 2:start] in ('""', "''") and text[end:end + 2] in ('""', "''")):
        return True

    # 2. Check preceding 50 characters / up to 5 preceding words for negation
    preceding_window = text[max(0, start - 60):start]
    # Stop at sentence boundary if any (. ! ?)
    sentence_break = max(preceding_window.rfind("."), preceding_window.rfind(";"), preceding_window.rfind("\n"))
    if sentence_break != -1:
        preceding_window = preceding_window[sentence_break + 1:]

    preceding_tokens = re.findall(r"[A-Za-z'\-]+", preceding_window.lower())
    for token in preceding_tokens[-5:]:
        clean_tok = token.replace("-", "")
        if clean_tok in NEGATION_WORDS or token in NEGATION_WORDS or token.startswith("non-") or token.startswith("non"):
            return True

    # 3. Check for immediate compound phrase like "does not prove safe" or "cannot be guaranteed"
    combined = (preceding_window + " " + text[start:end]).lower()
    if any(neg in combined for neg in ["does not", "do not", "will not", "cannot be", "not be", "is not", "was not", "never"]):
        return True

    return False


def check_banned_phrases(text: str, allow_disclaimers: bool = True) -> List[str]:
    """
    Scans text for occurrences of banned phrases.
    If allow_disclaimers is True, negated occurrences (e.g. 'not validated for GMP use')
    and quoted occurrences (e.g. 'validated') are treated as valid disclaimers and not flagged.
    Returns list of offending affirmative banned phrases detected.
    """
    affirmative_violations: List[str] = []

    for phrase, pattern in BANNED_PHRASE_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            if allow_disclaimers and _is_negated_or_quoted(text, start, end):
                continue
            # If not negated/quoted, it is an affirmative violation
            if phrase not in affirmative_violations:
                affirmative_violations.append(phrase)

    return affirmative_violations


def assert_no_banned_phrases(text: str, context: str = "", allow_disclaimers: bool = True) -> None:
    """
    Raises a ValueError if any un-negated, un-quoted banned phrase appears in text.
    """
    violations = check_banned_phrases(text, allow_disclaimers=allow_disclaimers)
    if violations:
        ctx_msg = f" in {context}" if context else ""
        raise ValueError(
            f"Banned phrase violation{ctx_msg}! Prohibited affirmative claims found: {violations}. "
            "Under Guardrail G2, outputs must never state or imply autonomous GMP disposition."
        )


def apply_draft_stamp(content: str) -> str:
    """
    Prepends the mandatory DRAFT stamp to an output string if not already present.
    """
    if DRAFT_STAMP in content:
        return content
    return f"[{DRAFT_STAMP}]\n\n{content}"


def scan_file_for_banned_phrases(file_path: Path, allow_disclaimers: bool = True) -> List[Tuple[int, str, str]]:
    """
    Scans a generated output file line by line for banned phrases.
    Returns a list of (line_number, banned_phrase, line_content).
    """
    violations: List[Tuple[int, str, str]] = []
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return violations

    for line_idx, line in enumerate(content.splitlines(), start=1):
        found = check_banned_phrases(line, allow_disclaimers=allow_disclaimers)
        for phrase in found:
            violations.append((line_idx, phrase, line.strip()))
    return violations


def check_doc_claims_hygiene(line: str) -> List[Tuple[str, str]]:
    """
    Strict linter for documentation (README.md, DEMO.md, LIMITATIONS.md, docs/*.md)
    and skill.md files:
    - guaranteed: allow only negations (e.g. 'cannot be guaranteed', 'not guaranteed') or in quotes.
    - 'proves safe': allow only negations or in quotes.
    - 'release for use': allow only negations or in quotes.
    - 'zero risk': allow only negations or in quotes.
    - validated: allow only negations like 'not validated', 'non-validated', 'unvalidated',
      'does not constitute a validated', or in quotes.
    - compliant: allow only 'not a compliant', 'not compliant', 'non-compliant', or in quotes.
    - immutable: allow only task-semantics like 'immutable fixed tasks' or 'immutable, hash-chained' or in quotes.
    Returns list of (violation_term, line_content).
    """
    violations: List[Tuple[str, str]] = []
    line_clean = line.strip()
    l_lower = line_clean.lower()

    # 1. guarantee / guaranteed
    if "guarantee" in l_lower:
        is_quoted = bool(re.search(r'["\'`]\s*guarantee(?:d|s)?\s*["\'`]', line_clean, re.IGNORECASE))
        is_negated = bool(re.search(r"\b(?:cannot|can\s+not|not|never|without)\s+(?:be\s+)?guarantee(?:d|s)?\b", l_lower))
        if not (is_quoted or is_negated):
            violations.append(("guaranteed", line_clean))

    # 2. proves safe / prove safe
    if "prove safe" in l_lower or "proves safe" in l_lower or "proved safe" in l_lower:
        is_quoted = bool(re.search(r'["\'`]\s*prove(?:d|s)?\s+safe\s*["\'`]', line_clean, re.IGNORECASE))
        is_negated = bool(re.search(r"\b(?:does\s+not|not|never|cannot)\s+prove(?:d|s)?\s+safe\b", l_lower))
        if not (is_quoted or is_negated):
            violations.append(("proves safe", line_clean))

    # 3. release for use
    if "release for use" in l_lower or "released for use" in l_lower:
        is_quoted = bool(re.search(r'["\'`]\s*release(?:d)?\s+for\s+use\s*["\'`]', line_clean, re.IGNORECASE))
        is_negated = bool(re.search(r"\b(?:never|not|prohibit(?:s)?)\s+release(?:d)?\s+for\s+use\b", l_lower))
        if not (is_quoted or is_negated):
            violations.append(("release for use", line_clean))

    # 4. zero risk
    if "zero risk" in l_lower:
        is_quoted = bool(re.search(r'["\'`]\s*zero\s+risk\s*["\'`]', line_clean, re.IGNORECASE))
        is_negated = bool(re.search(r"\b(?:not|never|does\s+not\s+imply|no)\s+zero\s+risk\b", l_lower))
        if not (is_quoted or is_negated):
            violations.append(("zero risk", line_clean))

    # 5. validated / validate
    if re.search(r"\bvalidate(?:d|s)?\b", l_lower):
        is_quoted = bool(re.search(r'["\'`]\s*validate(?:d|s)?\s*["\'`]', line_clean, re.IGNORECASE))
        is_negated = bool(re.search(r"\b(?:not\s+(?:a\s+)?validated|unvalidated|non-validated|non\s+validated|does\s+not\s+constitute\s+a\s+validated|not\s+yet\s+performed|without\s+re-validation)\b", l_lower))
        is_stats_context = bool(re.search(r"\b(?:validation\s+weeks|validation\s+lots|validation\s+split|validation\s+cohort|model_card_validation|cross-validation)\b", l_lower))
        if not (is_quoted or is_negated or is_stats_context):
            violations.append(("validated", line_clean))

    # 6. compliant
    if "compliant" in l_lower:
        is_quoted = bool(re.search(r'["\'`]\s*compliant\s*["\'`]', line_clean, re.IGNORECASE))
        is_allowed_neg = bool(re.search(r"\b(?:not\s+a\s+compliant|not\s+compliant|non-compliant|non\s+compliant|nor\s+compliant)\b", l_lower))
        if not (is_quoted or is_allowed_neg):
            violations.append(("compliant", line_clean))

    # 7. immutable
    if "immutable" in l_lower:
        is_quoted = bool(re.search(r'["\'`]\s*immutable\s*["\'`]', line_clean, re.IGNORECASE))
        is_allowed_semantics = bool(re.search(r"\b(?:immutable\s+fixed\s+tasks|immutable\s+tier-1a|immutable,\s+hash-chained|immutable\s+audit)\b", l_lower))
        if not (is_quoted or is_allowed_semantics):
            violations.append(("immutable", line_clean))

    # 8. extended terms: GMP-ready, audit-ready, inspection-ready, production-grade
    for term, pat in [
        ("GMP-ready", r"\bgmp[- ]ready\b"),
        ("audit-ready", r"\baudit[- ]ready\b"),
        ("inspection-ready", r"\binspection[- ]ready\b"),
        ("production-grade", r"\bproduction[- ]grade\b"),
    ]:
        if re.search(pat, l_lower):
            is_quoted = bool(re.search(r'["\'`].*' + pat + r'.*["\'`]', line_clean, re.IGNORECASE))
            is_neg = bool(re.search(r"\b(?:not|never|no|non|without)\s+(?:a\s+)?" + pat, l_lower))
            if not (is_quoted or is_neg):
                violations.append((term, line_clean))

    return violations


def scan_documentation_and_skills(base_dir: Optional[Path] = None) -> List[Tuple[str, int, str, str]]:
    """
    Scans README.md, DEMO.md, LIMITATIONS.md, docs/*.md, and all skill.md files for claims violations.
    Returns list of (file_path, line_number, violation_term, line_content).
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    target_files: List[Path] = [
        base_dir / "README.md",
        base_dir / "DEMO.md",
        base_dir / "LIMITATIONS.md",
    ]
    target_files.extend(base_dir.glob("docs/*.md"))
    target_files.extend(base_dir.glob("**/skill.md"))
    target_files.append(base_dir / "agents/agent_4_stakeholder_comms/assets/email_templates.json")
    target_files.append(base_dir / "app.py")
    target_files.append(base_dir / "pm/ui_view.py")

    all_violations: List[Tuple[str, int, str, str]] = []

    for file_path in target_files:
        if not file_path.is_file():
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line_idx, line in enumerate(content.splitlines(), start=1):
            line_viols = check_doc_claims_hygiene(line)
            for term, text in line_viols:
                all_violations.append((str(file_path.relative_to(base_dir)), line_idx, term, text))

    return all_violations
