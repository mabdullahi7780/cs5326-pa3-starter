"""
src/prm_judge.py - PRM judge prompt construction, parsing, and step scoring.
"""
import re
from typing import Any, Callable

from .utils import make_hf_judge_fn

JudgeFn = Callable[[str], dict]
# judge_fn(judge_prompt) -> {"text": str, "num_tokens": int}

_JUDGE_HEADER = "You are checking one step of a student's solution to a math word problem.\n\nProblem:\n"

# The verdict comes first so the 96-token cap can never cut it off, and the two
# verdict literals sit on separate lines so a judge that copies the template is
# not read as VALID by the first-match rule.
_JUDGE_INSTRUCTIONS = (
    "\n\n"
    "Judge only the Current Step. Assume all preceding steps are correct, exactly as written, "
    "even if one of them contains a mistake.\n"
    "The Current Step is VALID if its arithmetic is correct and it follows logically from the "
    "problem and the preceding steps.\n"
    "The Current Step is INVALID if it contains an arithmetic error, uses a number or fact that is "
    "not in the problem or the preceding steps, or does not follow from them.\n"
    "\n"
    "Reply with exactly two lines.\n"
    "Line 1 is either\n"
    "Verdict: VALID\n"
    "or\n"
    "Verdict: INVALID\n"
    "Line 2 is\n"
    "Rationale: <one sentence explaining your decision>"
)

# Rule 1 of the verdict parser, applied to one line at a time.
_VERDICT_LINE_RE = re.compile(r"verdict\s*:\s*\**\s*(VALID|INVALID)\b", re.IGNORECASE)
# Rules 2 and 3: standalone words only, so 'INVALID' never counts as 'VALID'.
_NEGATIVE_RE = re.compile(r"\binvalid\b|\bnot\s+valid\b", re.IGNORECASE)
_POSITIVE_RE = re.compile(r"\bvalid\b", re.IGNORECASE)
_RATIONALE_RE = re.compile(r"rationale\s*:", re.IGNORECASE)
# A sentence ends at . ! or ? followed by whitespace or the end, so '2.5' stays whole.
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")


def build_prm_judge_prompt(question: str, preceding_steps: list[str], current_step: str) -> str:
    """
    Builds the verification prompt for the PRM judge to classify current_step
    as VALID or INVALID given the problem question and preceding_steps.
    
    Judge Semantics (Local Validity):
    The judge evaluates current_step ASSUMING ALL PRECEDING STEPS ARE CORRECT.
    A step is VALID if its arithmetic is sound and it follows logically from the
    problem statement plus preceding steps as written. An error in an earlier step
    does NOT make subsequent steps invalid by itself.
    
    Autograder Literal Checks:
    - Preceding steps are rendered as 'Step 1: <text>', 'Step 2: <text>', ... in order.
    - The current step is rendered with the label 'Current Step:'.
    - The prompt contains the phrase 'assume all preceding steps are correct' (case-insensitive).
    - The prompt contains the literal strings 'Verdict: VALID', 'Verdict: INVALID', and 'Rationale:'.
    - With an empty preceding_steps list, the prompt states '(no preceding steps)'.
    """
    question_text = "" if question is None else str(question).strip()
    steps = [str(step).strip() for step in (preceding_steps or [])]
    if steps:
        preceding = "\n".join(f"Step {i}: {step}" for i, step in enumerate(steps, start=1))
    else:
        preceding = "(no preceding steps)"
    current = "" if current_step is None else str(current_step).strip()
    # Concatenation rather than str.format, so braces in a question or step are safe.
    return (
        _JUDGE_HEADER + question_text
        + "\n\nPreceding steps:\n" + preceding
        + "\n\nCurrent Step: " + current
        + _JUDGE_INSTRUCTIONS
    )


def parse_judge_decision(judge_output: str) -> tuple[float, str]:
    r"""
    Parses the judge model's raw text completion into (score, rationale).
    
    Rules:
      1. If a line matches /verdict\s*:\s*\**\s*(VALID|INVALID)\b/i, use it.
         If several exist, use the first.
      2. Otherwise, if standalone word INVALID or phrase 'NOT VALID' appears
         (case-insensitive, word boundaries) -> -1.0.
      3. Otherwise, if standalone word VALID appears -> +1.0.
      4. Otherwise -> -1.0 (unparseable defaults to INVALID).
      
    Warning: A naive '"VALID" in text' check is wrong because "INVALID" contains "VALID".
    
    Rationale: Text after 'Rationale:' up to the first sentence end (".", "!", or "?"
    followed by whitespace or end of string, so decimals like "2.5" are not split),
    or a newline, whichever comes first, stripped. If absent, returns empty string "".
    """
    if not isinstance(judge_output, str) or not judge_output:
        return -1.0, ""

    # Rule 1: the first line holding a verdict decides.
    score = None
    for line in judge_output.splitlines():
        match = _VERDICT_LINE_RE.search(line)
        if match:
            score = 1.0 if match.group(1).upper() == "VALID" else -1.0
            break
    if score is None:
        if _NEGATIVE_RE.search(judge_output):    # rule 2
            score = -1.0
        elif _POSITIVE_RE.search(judge_output):  # rule 3
            score = 1.0
        else:                                    # rule 4: unparseable counts as INVALID
            score = -1.0

    # Rationale: after the first 'Rationale:', up to the first sentence end or newline.
    rationale = ""
    label = _RATIONALE_RE.search(judge_output)
    if label:
        rest = judge_output[label.end():].lstrip(" \t")
        rest = rest.split("\n", 1)[0]
        end = _SENTENCE_END_RE.search(rest)
        if end:
            rest = rest[:end.end()]
        rationale = rest.strip()
    return score, rationale


def score_path(
    judge_model: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    candidate: dict[str, Any] | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Sequentially scores each step in candidate["steps"] using judge_fn.
    Step t is judged with preceding_steps = steps[:t].
    
    The student's code MUST call build_prm_judge_prompt to build the string passed to judge_fn,
    and MUST call parse_judge_decision on the returned text.
    
    Returns:
      step_scores, rationales, judge_calls, judge_tokens
    """
    steps = list((candidate or {}).get("steps") or [])
    if judge_fn is None and steps:
        judge_fn = make_hf_judge_fn(judge_model, judge_tokenizer)

    step_scores: list[float] = []
    rationales: list[str] = []
    judge_tokens = 0
    # Every step is judged (no early stop): judge calls are the sum of K_i.
    for t, step in enumerate(steps):
        prompt = build_prm_judge_prompt(question, steps[:t], step)
        output = judge_fn(prompt)
        score, rationale = parse_judge_decision(output["text"])
        step_scores.append(score)
        rationales.append(rationale)
        judge_tokens += int(output["num_tokens"])

    return {
        "step_scores": step_scores,
        "rationales": rationales,
        "judge_calls": len(steps),
        "judge_tokens": judge_tokens,
    }


def judge_prompts_for_path(question: str, steps: list[str]) -> list[str]:
    """The judge prompts score_path builds for one path, in the order it calls judge_fn."""
    steps = list(steps or [])
    return [build_prm_judge_prompt(question, steps[:t], step) for t, step in enumerate(steps)]
