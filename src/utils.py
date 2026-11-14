"""
src/utils.py - Model loading, generation constants, seed management, and factories.
"""
import json
import math
import os
from typing import Any, Callable

# Global random seed specified by course policy
GLOBAL_SEED = 5326

# Sampling & decoding hyperparameter configurations
GREEDY_GEN_KWARGS = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 512,
}

SAMPLING_GEN_KWARGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512,
}

JUDGE_GEN_KWARGS = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 96,
}


def set_seed_for_problem(problem_index: int) -> None:
    """
    Sets a deterministic seed for problem_index (5326 + problem_index).
    Guarantees reproducibility and safe resumability after interruptions.
    """
    try:
        import transformers
        transformers.set_seed(GLOBAL_SEED + problem_index)
    except ImportError:
        import random
        random.seed(GLOBAL_SEED + problem_index)


def load_models(generator_device: str = "cuda:0", judge_device: str = "cuda:1"):
    """
    Loads Qwen2.5-1.5B-Instruct in float16 (Generator) and
    Qwen2.5-7B-Instruct in 4-bit NF4 (Judge).
    
    Hardware Mapping:
      - Kaggle GPU T4x2: generator on cuda:0, judge on cuda:1.
      - Single GPU: both models placed on cuda:0.
      - Warning: Never use device_map="auto" across two GPUs for a single model!
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus < 2:
        g_dev = "cuda:0" if num_gpus == 1 else "cpu"
        j_dev = "cuda:0" if num_gpus == 1 else "cpu"
    else:
        g_dev = generator_device
        j_dev = judge_device

    # 1. Generator: Qwen/Qwen2.5-1.5B-Instruct in float16 (~3.1 GB VRAM)
    gen_id = "Qwen/Qwen2.5-1.5B-Instruct"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_id)
    if gen_tokenizer.pad_token_id is None:
        gen_tokenizer.pad_token_id = gen_tokenizer.eos_token_id

    gen_device_map = {"": g_dev} if g_dev != "cpu" else None
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_id,
        torch_dtype=torch.float16 if g_dev != "cpu" else torch.float32,
        device_map=gen_device_map,
    )
    if g_dev == "cpu":
        gen_model = gen_model.to("cpu")

    # 2. Judge: Qwen/Qwen2.5-7B-Instruct in 4-bit NF4 (~5.5 GB VRAM)
    judge_id = "Qwen/Qwen2.5-7B-Instruct"
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_id)
    if judge_tokenizer.pad_token_id is None:
        judge_tokenizer.pad_token_id = judge_tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    judge_device_map = {"": j_dev} if j_dev != "cpu" else None
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_id,
        quantization_config=bnb_config if j_dev != "cpu" else None,
        torch_dtype=torch.float16 if j_dev != "cpu" else torch.float32,
        device_map=judge_device_map,
    )
    if j_dev == "cpu":
        judge_model = judge_model.to("cpu")

    return gen_model, gen_tokenizer, judge_model, judge_tokenizer


def _get_eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    """Extracts all EOS token IDs from model.generation_config and tokenizer."""
    eos_ids: set[int] = set()
    if hasattr(model, "generation_config") and model.generation_config is not None:
        cfg_eos = getattr(model.generation_config, "eos_token_id", None)
        if cfg_eos is not None:
            if isinstance(cfg_eos, int):
                eos_ids.add(cfg_eos)
            elif isinstance(cfg_eos, (list, tuple, set)):
                eos_ids.update(cfg_eos)
    if tokenizer is not None:
        tok_eos = getattr(tokenizer, "eos_token_id", None)
        if tok_eos is not None:
            if isinstance(tok_eos, int):
                eos_ids.add(tok_eos)
            elif isinstance(tok_eos, (list, tuple, set)):
                eos_ids.update(tok_eos)
    return eos_ids


def _slice_to_first_eos(gen_tokens: Any, eos_ids: set[int]) -> Any:
    """Slices token sequence up to and including the first EOS token."""
    if hasattr(gen_tokens, "tolist"):
        ids = gen_tokens.tolist()
    else:
        ids = list(gen_tokens)
    first_eos_idx = None
    for idx, tok in enumerate(ids):
        if tok in eos_ids:
            first_eos_idx = idx
            break
    if first_eos_idx is not None:
        return gen_tokens[:first_eos_idx + 1]
    return gen_tokens


def make_hf_generate_fn(model: Any, tokenizer: Any) -> Callable[[str, int, dict], list[dict]]:
    """
    Factory creating a GenerateFn conforming to submission_adapter contract.
    Applies the Qwen chat template to raw instruction prompts.
    """
    def generate_fn(prompt: str, num_samples: int, gen_kwargs: dict) -> list[dict]:
        try:
            import torch
            no_grad_ctx = torch.no_grad
        except ImportError:
            class _NoGrad:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            no_grad_ctx = lambda: _NoGrad()

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        kwargs = dict(gen_kwargs)
        if not kwargs.get("do_sample", False):
            kwargs["temperature"] = None
            kwargs["top_p"] = None
            kwargs["top_k"] = None

        if num_samples > 1 and kwargs.get("do_sample", True):
            kwargs["num_return_sequences"] = num_samples

        kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

        with no_grad_ctx():
            output_ids = model.generate(**inputs, **kwargs)

        eos_ids = _get_eos_token_ids(model, tokenizer)
        results = []
        for i in range(output_ids.shape[0]):
            gen_tokens = output_ids[i][prompt_len:]
            gen_tokens_valid = _slice_to_first_eos(gen_tokens, eos_ids)
            text = tokenizer.decode(gen_tokens_valid, skip_special_tokens=True)
            results.append({"text": text, "num_tokens": len(gen_tokens_valid)})
        return results

    return generate_fn


def make_hf_judge_fn(judge_model: Any, judge_tokenizer: Any) -> Callable[[str], dict]:
    """
    Factory creating a JudgeFn conforming to submission_adapter contract.
    Applies the Qwen chat template to raw judge prompts.
    """
    def judge_fn(judge_prompt: str) -> dict:
        try:
            import torch
            no_grad_ctx = torch.no_grad
        except ImportError:
            class _NoGrad:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            no_grad_ctx = lambda: _NoGrad()

        messages = [{"role": "user", "content": judge_prompt}]
        formatted = judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = judge_tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(judge_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        kwargs = dict(JUDGE_GEN_KWARGS)
        kwargs.setdefault("pad_token_id", judge_tokenizer.eos_token_id)

        with no_grad_ctx():
            output_ids = judge_model.generate(**inputs, **kwargs)

        eos_ids = _get_eos_token_ids(judge_model, judge_tokenizer)
        gen_tokens = output_ids[0][prompt_len:]
        gen_tokens_valid = _slice_to_first_eos(gen_tokens, eos_ids)
        text = judge_tokenizer.decode(gen_tokens_valid, skip_special_tokens=True)
        return {"text": text, "num_tokens": len(gen_tokens_valid)}

    return judge_fn


def verdict_token_pairs(tokenizer: Any) -> tuple[tuple[int, int], ...]:
    """
    Token ids of the two verdict words, as (valid_id, invalid_id) pairs, for the
    optional product_logprob bonus (manual 2.4, item 4).

    The first pair is the form after 'Verdict:' (' VALID', ' INVALID'); the second
    is the form with no leading space (as after '**'). Each word must be a single
    token, so the first-token logits are the full-sequence comparison.
    """
    pairs = []
    for valid_text, invalid_text in ((" VALID", " INVALID"), ("VALID", "INVALID")):
        valid_ids = tokenizer.encode(valid_text, add_special_tokens=False)
        invalid_ids = tokenizer.encode(invalid_text, add_special_tokens=False)
        if len(valid_ids) != 1 or len(invalid_ids) != 1:
            raise ValueError(f"verdict words are not single tokens: {valid_ids} {invalid_ids}")
        pairs.append((int(valid_ids[0]), int(invalid_ids[0])))
    return tuple(pairs)


def _two_way_softmax(z_valid: float, z_invalid: float) -> float:
    """exp(z_valid) / (exp(z_valid) + exp(z_invalid)), written so it cannot overflow."""
    diff = z_invalid - z_valid
    if diff >= 0:
        e = math.exp(-diff)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(diff))


def _verdict_probability(gen_ids: Any, step_logits: Any, row: int, pairs: tuple) -> float | None:
    """
    P(VALID) at the first generated verdict token, from the raw logits of that step.

    step_logits[t] holds the logits used to choose generated token t. Returns None
    when no verdict token was generated.
    """
    ids = gen_ids.tolist() if hasattr(gen_ids, "tolist") else list(gen_ids)
    for t, token in enumerate(ids):
        for valid_id, invalid_id in pairs:
            if token == valid_id or token == invalid_id:
                z_valid = float(step_logits[t][row][valid_id])
                z_invalid = float(step_logits[t][row][invalid_id])
                return _two_way_softmax(z_valid, z_invalid)
    return None


def judge_prompts_batched(
    judge_model: Any, judge_tokenizer: Any, prompts: list[str], batch_size: int = 16,
    return_verdict_probs: bool = False
) -> list[dict]:
    """
    Runs many judge prompts through batched greedy generate() calls.

    Uses the same chat template, JUDGE_GEN_KWARGS, EOS slicing and token counting
    as make_hf_judge_fn, with left padding so every row's new tokens start at the
    same column. Returns one {"text", "num_tokens"} per prompt, in input order.
    Every prompt is generated, duplicates included, so the token counts are real.

    With return_verdict_probs=True (the optional bonus), generate() also returns the
    raw logits (output_logits, before the repetition penalty and other processors),
    and each result gets "verdict_prob": P(VALID) at the first generated verdict
    token, or None if there is none. The generated tokens are the same either way.
    """
    try:
        import torch
        no_grad_ctx = torch.no_grad
    except ImportError:
        class _NoGrad:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        no_grad_ctx = lambda: _NoGrad()

    prompts = list(prompts or [])
    if not prompts:
        return []
    batch_size = max(1, int(batch_size))
    eos_ids = _get_eos_token_ids(judge_model, judge_tokenizer)
    kwargs = dict(JUDGE_GEN_KWARGS)
    kwargs.setdefault("pad_token_id", judge_tokenizer.eos_token_id)
    pairs = ()
    if return_verdict_probs:
        pairs = verdict_token_pairs(judge_tokenizer)
        kwargs["output_logits"] = True
        kwargs["return_dict_in_generate"] = True

    previous_side = getattr(judge_tokenizer, "padding_side", None)
    judge_tokenizer.padding_side = "left"
    results: list[dict] = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            formatted = [
                judge_tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
                )
                for prompt in chunk
            ]
            inputs = judge_tokenizer(formatted, return_tensors="pt", padding=True)
            inputs = {k: v.to(judge_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            with no_grad_ctx():
                generated = judge_model.generate(**inputs, **kwargs)
            output_ids = generated.sequences if return_verdict_probs else generated

            for i in range(output_ids.shape[0]):
                gen_tokens_valid = _slice_to_first_eos(output_ids[i][prompt_len:], eos_ids)
                text = judge_tokenizer.decode(gen_tokens_valid, skip_special_tokens=True)
                record = {"text": text, "num_tokens": len(gen_tokens_valid)}
                if return_verdict_probs:
                    record["verdict_prob"] = _verdict_probability(gen_tokens_valid, generated.logits, i, pairs)
                results.append(record)
            del generated  # the raw logits are large; free them before the next batch
    finally:
        if previous_side is not None:
            judge_tokenizer.padding_side = previous_side
    return results


def make_replay_judge_fn(prompts: list[str], outputs: list[dict]) -> Callable[[str], dict]:
    """
    A JudgeFn that hands back precomputed judge outputs in order.

    score_path calls it exactly as it would call the real judge. Each call must
    bring the next expected prompt, so outputs from a batched pre-pass can never
    be matched to the wrong step.
    """
    prompts = list(prompts)
    outputs = list(outputs)
    if len(prompts) != len(outputs):
        raise ValueError(f"{len(prompts)} prompts but {len(outputs)} outputs")
    position = [0]

    def judge_fn(judge_prompt: str) -> dict:
        index = position[0]
        if index >= len(prompts):
            raise IndexError("more judge calls than precomputed outputs")
        if judge_prompt != prompts[index]:
            raise ValueError(f"judge call {index} does not match the precomputed prompt")
        position[0] += 1
        return dict(outputs[index])

    judge_fn.remaining = lambda: len(prompts) - position[0]  # type: ignore[attr-defined]
    return judge_fn


def save_results_summary(results_list: list[dict[str, Any]], filepath: str = "outputs/results_summary.json") -> None:
    """
    Saves results_summary.json conforming strictly to the assignment schema:
    Keys: 'Method', 'Accuracy (%)', 'Generation Calls', 'Judge Calls', 'Total LLM Calls',
          'Avg Gen Tokens / Problem', 'Avg Judge Tokens / Problem', 'Total Time (s)'
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)


def save_aggregation_ablation(ablation_list: list[dict[str, Any]], filepath: str = "outputs/aggregation_ablation.json") -> None:
    """
    Saves aggregation_ablation.json:
    List of {'Strategy': str, 'Lambda': float | None, 'Accuracy (%)': float, 'Tie Rate (%)': float}
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(ablation_list, f, indent=2)


def save_sc_scaling_curve(curve_list: list[dict[str, Any]], filepath: str = "outputs/sc_scaling_curve.json") -> None:
    """
    Saves sc_scaling_curve.json:
    List of {'N': int, 'Accuracy (%)': float, 'Avg Gen Tokens / Problem': float}
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(curve_list, f, indent=2)
