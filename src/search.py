"""
src/search.py - Path aggregation, Best-of-N search, and compute-matched search.
"""
import time
from typing import Any, Callable

from .generation import compute_majority_vote, run_self_consistency
from .prm_judge import score_path

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]

# 'product_logprob' is the optional bonus (manual 2.4, item 4): its step scores are
# the judge's verdict probabilities p_t in [0, 1], not +1/-1 verdicts.
AGGREGATION_STRATEGIES = ("min", "product", "step_penalty", "product_logprob")

# Path scores within this distance of the best one count as tied (manual 5.3, rule 3).
# Float products depend on the order of the steps, so exact equality is not safe.
TIE_TOLERANCE = 1e-9


def aggregate_path_score(
    step_scores: list[float], 
    strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> float:
    """
    Aggregates a list of step scores (+1.0 or -1.0) into a scalar trajectory score.
    
    Rules:
      - An empty list returns -1.0 for every strategy.
      - 'min': min(step_scores).
      - 'product': p_t = clamp((r_t + 1)/2, 0.01, 0.99); returns the raw product.
      - 'step_penalty': mean(step_scores) - penalty_lambda * count(r_t == -1.0).
      - An unknown strategy raises ValueError.
    """
    # Checked first, so a misspelt strategy is caught even for an empty path.
    if strategy not in AGGREGATION_STRATEGIES:
        raise ValueError(f"Unknown aggregation strategy: {strategy!r}")

    scores = [float(score) for score in (step_scores or [])]
    if not scores:
        return -1.0

    if strategy == "min":
        return min(scores)

    if strategy == "product":
        product = 1.0
        for r in scores:
            product *= min(max((r + 1.0) / 2.0, 0.01), 0.99)
        return product

    if strategy == "step_penalty":
        num_invalid = sum(1 for r in scores if r == -1.0)
        return sum(scores) / len(scores) - float(penalty_lambda) * num_invalid

    # product_logprob: the joint validity of the path, from continuous probabilities.
    product = 1.0
    for p in scores:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"product_logprob expects probabilities in [0, 1], got {p}")
        product *= p
    return product


def select_best_path(
    candidate_paths: list[dict[str, Any]], 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> dict[str, Any]:
    """
    Selects the winning candidate path from candidate_paths.
    
    Input Dictionary Contract:
      Each candidate dictionary in candidate_paths must contain 'step_scores' (list[float])
      and 'answer' (int | None). Any other keys present in the candidate dictionary
      are passed through unchanged in the returned copy.
    
    Selection & Tie-Breaking Rules:
      1. Compute path_score for each candidate with aggregate_path_score.
         A candidate with an empty step_scores list gets -1.0.
      2. Eligibility: if at least one candidate has a non-None answer, candidates
         with answer=None are excluded. If all answers are None, all remain eligible.
      3. Find maximum path_score among eligible candidates (tolerance 1e-9).
         The tied set is all eligible candidates within tolerance of the maximum.
      4. If the tied set has more than one candidate, compute compute_majority_vote
         over their answers. Return the lowest-index candidate in the tied set whose
         answer equals the vote. If the vote is None, return the lowest-index tied candidate.
      5. Returns a shallow COPY of the chosen dict with two added keys:
         'path_score' (float) and 'num_tied' (int, size of the tied set).
         Does not mutate the inputs.
    """
    candidates = list(candidate_paths or [])
    if not candidates:
        raise ValueError("select_best_path needs at least one candidate")

    # Rule 1: one path score per candidate. A missing or empty step list scores -1.0.
    scores = [
        aggregate_path_score(candidate.get("step_scores") or [], aggregation_strategy, penalty_lambda)
        for candidate in candidates
    ]
    answers = [candidate.get("answer") for candidate in candidates]

    # Rule 2: drop answer=None candidates, unless every answer is None.
    if any(answer is not None for answer in answers):
        eligible = [i for i, answer in enumerate(answers) if answer is not None]
    else:
        eligible = list(range(len(candidates)))

    # Rule 3: the tied set is every eligible candidate within the tolerance of the best score.
    best = max(scores[i] for i in eligible)
    tied = [i for i in eligible if scores[i] >= best - TIE_TOLERANCE]

    # Rule 4: vote among the tied answers and take the lowest index with the voted answer.
    # After rule 2 the vote is None only when every tied answer is None.
    chosen = tied[0]
    if len(tied) > 1:
        vote = compute_majority_vote([answers[i] for i in tied])
        if vote is not None:
            chosen = next(i for i in tied if answers[i] == vote)

    # Rule 5: a shallow copy, so the input dicts are never changed.
    selected = dict(candidates[chosen])
    selected["path_score"] = float(scores[chosen])
    selected["num_tied"] = len(tied)
    return selected


def run_prm_best_of_n(
    model: Any = None, 
    judge_model: Any = None, 
    gen_tokenizer: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    n: int = 5, 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0, 
    candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Executes PRM-guided Best-of-N search over reasoning trajectories.
    
    If candidates is provided (shared candidate pool), does NOT sample new candidates
    (generation_calls = 0, generated_tokens = 0). Provided candidates use the
    run_self_consistency candidate format (at minimum 'steps' and 'answer').
    Otherwise, samples n candidates using run_self_consistency.
    
    Scores each candidate's steps using score_path.
    Selects the winning candidate via select_best_path.
    
    Returns:
      selected, answer, candidates [each augmented with step_scores, rationales],
      generation_calls, judge_calls, generated_tokens, judge_tokens, time_seconds
    """
    start = time.perf_counter()

    if candidates is None:
        # No shared pool: sample n paths, and their cost belongs to this call.
        sampled = run_self_consistency(model, gen_tokenizer, question, n=n, generate_fn=generate_fn)
        pool = sampled["candidates"]
        generation_calls = sampled["generation_calls"]
        generated_tokens = sampled["generated_tokens"]
    else:
        # Shared pool: judge every candidate passed in and sample nothing.
        pool = list(candidates)
        generation_calls = 0
        generated_tokens = 0

    judged = []
    judge_calls = 0
    judge_tokens = 0
    for candidate in pool:
        result = score_path(judge_model, judge_tokenizer, question, candidate, judge_fn)
        # A new dict per candidate, so the shared pool is never changed.
        judged.append({
            **candidate,
            "step_scores": result["step_scores"],
            "rationales": result["rationales"],
        })
        judge_calls += result["judge_calls"]
        judge_tokens += result["judge_tokens"]

    selected = select_best_path(judged, aggregation_strategy, penalty_lambda) if judged else None

    return {
        "selected": selected,
        "answer": selected.get("answer") if selected is not None else None,
        "candidates": judged,
        "generation_calls": generation_calls,
        "judge_calls": judge_calls,
        "generated_tokens": generated_tokens,
        "judge_tokens": judge_tokens,
        "time_seconds": time.perf_counter() - start,
    }


def run_compute_matched_self_consistency(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    target_calls: int = 15, 
    temperature: float = 0.7, 
    existing_candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes compute-matched Self-Consistency aggregating target_calls paths.
    If existing_candidates are supplied, samples only the deficit:
    target_calls - len(existing_candidates). Returned candidates = existing_candidates followed by new samples, in sampling order.
    
    Returns:
      candidates, answer, generation_calls (new samples only), total_samples,
      generated_tokens (new samples only), time_seconds
    """
    start = time.perf_counter()
    existing = list(existing_candidates or [])

    # Sample only the deficit. A pool that already meets the target is returned whole.
    deficit = max(0, target_calls - len(existing))
    if deficit > 0:
        sampled = run_self_consistency(
            model, tokenizer, question, n=deficit, temperature=temperature, generate_fn=generate_fn
        )
        new_candidates = sampled["candidates"]
        generation_calls = sampled["generation_calls"]
        generated_tokens = sampled["generated_tokens"]
    else:
        new_candidates = []
        generation_calls = 0
        generated_tokens = 0

    # Existing first, then new, in sampling order. The vote breaks ties by earliest index,
    # so this order keeps N = target_calls on the scaling curve equal to this answer.
    all_candidates = existing + new_candidates
    answer = compute_majority_vote([candidate.get("answer") for candidate in all_candidates])

    return {
        "candidates": all_candidates,
        "answer": answer,
        "generation_calls": generation_calls,
        "total_samples": len(all_candidates),
        "generated_tokens": generated_tokens,
        "time_seconds": time.perf_counter() - start,
    }
