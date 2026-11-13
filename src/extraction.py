"""
src/extraction.py - Step and answer extraction utilities.
"""
import re


def parse_steps(completion_text: str) -> list[str]:
    """
    Parses completion text and returns a list of individual step strings.
    Must tolerate markdown decoration around labels (e.g. '**Step 1:**', '### Step 2:')
    and strip leftover '*' or '#' from captured text.
    """
    if not completion_text or not isinstance(completion_text, str):
        return []

    # Pattern matching Step markers with optional markdown decoration (#, *, `, :, .)
    pattern = r"(?:^|\n)\s*(?:[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)(.*?)(?=(?:\n\s*[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)|(?:\n\s*[#*`\s]*Final Answer)|$)"
    matches = re.findall(pattern, completion_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = []
    for m in matches:
        text = m.strip().strip("*#`").strip()
        if text:
            cleaned.append(text)
    return cleaned


# 'final answer' in any case, with any spacing between the two words.
_DELIMITER_RE = re.compile(r"final\s*answer", re.IGNORECASE)

# Characters allowed between the delimiter and the number: the colon,
# whitespace (including a line break), markdown marks and a currency sign.
# The currency sign may be LaTeX-escaped ('\$26'); a backslash is skipped only
# directly before '$', so '\boxed{18}' and '\(18\)' still return None.
_SKIP_RE = re.compile(r"^(?:[\s:*_`#$]|\\\$)*")

# Optional minus sign, optional (possibly escaped) currency sign, then digits
# (plain or grouped by thousands commas) and an optional decimal part.
_NUMBER_RE = re.compile(r"(-)?\s*(?:\\?\$)?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")

# A number directly followed by more digits, a comma and a digit, or a slash
# and a digit is a malformed grouping or a fraction, not an integer answer.
_MALFORMED_TAIL_RE = re.compile(r",?\d|\s*/\s*\d")


def extract_answer(completion_text: str) -> int | None:
    """
    Extracts the final numeric integer from the completion.

    Required Behaviors:
      - 'Final Answer: 18' -> 18
      - 'Final Answer: $1,234' -> 1234
      - 'Final Answer: 18.00' -> 18
      - 'Final Answer: 18.5' -> None (non-integer)
      - 'Final Answer: -5' -> -5
      - 'Final Answer: 18 dollars.' -> 18
      - '**Final Answer:** 18' -> 18
      - 'final answer 18' (case-insensitive, colon optional) -> 18
      - Multiple 'Final Answer' lines -> use the LAST one
      - No 'Final Answer' present -> None
      - Empty string or None-like garbage -> None, and never raise
    """
    if not isinstance(completion_text, str) or not completion_text:
        return None
    try:
        # Use the last delimiter; earlier ones are scratch attempts.
        delimiters = list(_DELIMITER_RE.finditer(completion_text))
        if not delimiters:
            return None
        rest = completion_text[delimiters[-1].end():]
        rest = _SKIP_RE.sub("", rest, count=1)

        # The answer must come first; there is no fallback to other integers.
        match = _NUMBER_RE.match(rest)
        if match is None:
            return None
        if _MALFORMED_TAIL_RE.match(rest[match.end():]):
            return None

        sign, digits, fraction = match.group(1), match.group(2), match.group(3)
        # Integer-valued decimals ('18.00') pass; others ('18.5') fail.
        if fraction is not None and fraction.strip("0"):
            return None
        value = int(digits.replace(",", ""))
        return -value if sign else value
    except Exception:
        return None
