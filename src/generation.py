"""
src/generation.py - Prompt formatting and baseline generation routines.
"""
import time
from typing import Any, Callable

from .extraction import extract_answer, parse_steps
from .utils import GREEDY_GEN_KWARGS, SAMPLING_GEN_KWARGS, make_hf_generate_fn

GenerateFn = Callable[[str, int, dict], list[dict]]
# generate_fn(raw_prompt, num_samples, gen_kwargs) -> list of {"text": str, "num_tokens": int}
# gen_kwargs carries do_sample/temperature/top_p/top_k/repetition_penalty/max_new_tokens

# Zero-shot instruction. The template lines follow the step schema of
# manual section 4.2; the rules guard against the markdown, LaTeX and
# \boxed{} drift described in its format compliance warning.
_PROMPT_HEADER = "Solve the following math problem step by step.\n\nQuestion: "
_PROMPT_FORMAT = (
    "\n\n"
    "Write your solution in exactly this format:\n"
    "Step 1: <first deduction or arithmetic operation>\n"
    "Step 2: <next deduction, building on the previous steps>\n"
    "...\n"
    "Step K: <final calculation>\n"
    "Final Answer: <integer>\n"
    "\n"
    "Rules:\n"
    '- Start every step on a new line with "Step k:", numbering from 1.\n'
    "- Each step makes one deduction or one calculation.\n"
    "- Use plain text only: no markdown headings, no bold, no LaTeX, no \\boxed{}.\n"
    '- The last line must be "Final Answer: " followed by the integer only, '
    "with no units, dollar sign or brackets.\n"
    "- Write nothing after the Final Answer line."
)


def format_step_prompt(question: str) -> str:
    """
    Formats the raw problem question into a prompt that instructs the
    model to emit numbered steps ('Step 1: ...') and 'Final Answer: <int>'.

    Returns a raw instruction string (no chat template applied).
    """
    # Concatenation rather than str.format, so braces in a question are safe.
    return _PROMPT_HEADER + question.strip() + _PROMPT_FORMAT


def compute_majority_vote(answers: list[int | None]) -> int | None:
    """
    Given a list of extracted answers, returns the majority answer.
    Ignores None values. Breaks ties deterministically in favor of the answer
    that appeared earliest in the list.
    """
    counts: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    for index, answer in enumerate(answers or []):
        if answer is None:  # 'is None', so an answer of 0 still gets its vote
            continue
        if answer not in counts:
            counts[answer] = 0
            first_seen[answer] = index
        counts[answer] += 1
    if not counts:
        return None
    # Highest count first; among equal counts, the earliest first appearance.
    return min(counts, key=lambda answer: (-counts[answer], first_seen[answer]))


def make_candidate(output: dict) -> dict[str, Any]:
    """Turns one generate_fn output into a candidate: completion, steps, answer, num_tokens."""
    completion = output["text"]
    return {
        "completion": completion,
        "steps": parse_steps(completion),
        "answer": extract_answer(completion),
        "num_tokens": int(output["num_tokens"]),
    }


def run_greedy(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes greedy decoding for a single question.
    Calls generate_fn with num_samples=1 and gen_kwargs containing do_sample=False,
    temperature=None, top_p=None, top_k=None.

    Returns:
      prompt, completion, steps, answer, generation_calls, generated_tokens, time_seconds
    """
    if generate_fn is None:
        generate_fn = make_hf_generate_fn(model, tokenizer)

    start = time.perf_counter()
    prompt = format_step_prompt(question)
    outputs = generate_fn(prompt, 1, dict(GREEDY_GEN_KWARGS))
    candidate = make_candidate(outputs[0])
    time_seconds = time.perf_counter() - start

    return {
        "prompt": prompt,
        "completion": candidate["completion"],
        "steps": candidate["steps"],
        "answer": candidate["answer"],
        "generation_calls": 1,
        "generated_tokens": candidate["num_tokens"],
        "time_seconds": time_seconds,
    }


def run_self_consistency(
    model: Any = None,
    tokenizer: Any = None,
    question: str = "",
    n: int = 5,
    temperature: float = 0.7,
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Samples n diverse reasoning trajectories and aggregates candidate answers via majority vote.

    Returns:
      candidates [each: completion, steps, answer, num_tokens], answer,
      generation_calls, generated_tokens, time_seconds
    """
    if generate_fn is None:
        generate_fn = make_hf_generate_fn(model, tokenizer)

    start = time.perf_counter()
    prompt = format_step_prompt(question)
    gen_kwargs = dict(SAMPLING_GEN_KWARGS)
    gen_kwargs["temperature"] = temperature
    # One batched call; each returned sequence counts as one generation call.
    outputs = generate_fn(prompt, n, gen_kwargs) if n > 0 else []
    candidates = [make_candidate(output) for output in outputs]
    answer = compute_majority_vote([candidate["answer"] for candidate in candidates])
    time_seconds = time.perf_counter() - start

    return {
        "candidates": candidates,
        "answer": answer,
        "generation_calls": len(candidates),
        "generated_tokens": sum(candidate["num_tokens"] for candidate in candidates),
        "time_seconds": time_seconds,
    }
