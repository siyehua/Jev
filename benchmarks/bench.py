#!/usr/bin/env python3
"""Benchmark: direct label-logit scoring vs chat generation (SPEC.md §13.9, §14).

Compares, on identical model weights, prompts, and hardware:
  1. direct label-logit scoring (one forward pass, no decode, 0 gen tokens)
  2. chat normal mode (answer directly, max_new_tokens=K)
  3. chat think mode (reason step by step, then answer; max_new_tokens=K)
  4. one-token label generation (kept in the raw JSON for SPEC.md §13.9
     compliance, excluded from the HTML report)

Reports cold (first call after load) and warm latencies, p50/p95, prompt
and generated token counts per mode, and memory use. Results are printed
as a table and written to JSON with --output.

Usage:
    python benchmarks/bench.py [--model Qwen/Qwen2.5-0.5B]
                                [--runs 20] [--think-tokens 256]
                                [--gen-tokens 32] [--output results.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev.prompt import PromptBuilder  # noqa: E402
from jev.scoring import softmax  # noqa: E402

CRITERIA = {
    "shipping": "A delivery, shipment, or package-arrival problem.",
    "billing": "A payment, invoice, or duplicate-charge problem.",
    "returns": "A return, refund, or exchange request.",
}
INPUT_TEXT = "My package arrived two weeks late, and my card was charged twice."
INSTRUCTIONS = "Choose the category that best matches the customer's main issue."

ANSWER_TAIL = "Respond with the letter of the best label only:\nAnswer:"
THINK_TAIL = (
    "Think step by step about which label best matches the context, then "
    "conclude with a final line formatted exactly as "
    "'Final answer: <letter>'.\nReasoning:"
)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _memory_mb() -> dict:
    result = {"cpu_rss_mb": _cpu_rss_mb()}
    if torch.cuda.is_available():
        result["cuda_allocated_mb"] = round(
            torch.cuda.max_memory_allocated() / 1e6, 2
        )
    return result


def _cpu_rss_mb() -> float:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 2)
    except OSError:
        pass
    return float("nan")


def _parse_think_answer(text: str, keys: tuple[str, ...]) -> str:
    """Extract the final label from a free-form reasoning answer.

    Fallback chain (conclusions tend to appear at the end of the text):
    1. explicit 'Final answer: <letter>'
    2. last '<letter>. <word>' pattern (e.g. 'B. billing')
    3. last '<key>' keyword mention (e.g. 'billing')
    4. raw text preview
    """
    match = re.search(r"final answer:\s*([A-C])", text, re.IGNORECASE)
    if match:
        return keys[ord(match.group(1).upper()) - ord("A")]
    letter_matches = list(re.finditer(r"\b([A-C])[.:]\s+\w", text))
    if letter_matches:
        return keys[ord(letter_matches[-1].group(1)) - ord("A")]
    for key in sorted(keys, key=len, reverse=True):
        keyword_matches = list(
            re.finditer(rf"\b{re.escape(key)}\b", text, re.IGNORECASE)
        )
        if keyword_matches:
            return key
    return text.strip()[:40]


def measure(fn: Callable[[], dict], runs: int) -> dict:
    """First call is 'cold', the remaining runs-1 are 'warm'.

    ``fn`` returns {"preview": str, "generated_tokens": int,
    "prompt_tokens": int}.
    """
    timings: list[float] = []
    cold_ms: float | None = None
    result: dict | None = None
    for i in range(runs):
        started = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - started) * 1000
        if i == 0:
            cold_ms = elapsed
        else:
            timings.append(elapsed)
    return {
        "cold_ms": round(cold_ms, 3),
        "warm_mean_ms": round(statistics.mean(timings), 3) if timings else None,
        "warm_p50_ms": round(_percentile(timings, 0.50), 3) if timings else None,
        "warm_p95_ms": round(_percentile(timings, 0.95), 3) if timings else None,
        "result_preview": str(result["preview"])[:60],
        "generated_tokens": result["generated_tokens"],
        "prompt_tokens": result["prompt_tokens"],
    }


def run(
    model_id: str,
    runs: int,
    gen_tokens: int,
    think_tokens: int,
    use_chat_template: bool,
) -> dict:
    print(f"loading {model_id} ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    print(f"device={device} dtype={dtype}")

    built = PromptBuilder("letters").build(INSTRUCTIONS, INPUT_TEXT, CRITERIA)
    prompt = built.prompt
    if use_chat_template and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    add_special = not (use_chat_template and tokenizer.chat_template)
    prompt_ids = tokenizer(prompt, add_special_tokens=add_special).input_ids
    input_tensor = torch.tensor([prompt_ids], device=device)

    # Think-mode prompt: same scenario, but the model must reason first
    # and then give a final answer line.
    think_prompt = built.prompt.replace(ANSWER_TAIL, THINK_TAIL)
    if use_chat_template and tokenizer.chat_template:
        think_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": think_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    think_ids = tokenizer(think_prompt, add_special_tokens=add_special).input_ids
    think_tensor = torch.tensor([think_ids], device=device)

    def direct_logits() -> dict:
        with torch.inference_mode():
            logits = model(
                input_ids=input_tensor, attention_mask=torch.ones_like(input_tensor)
            ).logits[0, -1]
        # Letter labels A/B/C in the exact continuation context.
        label_ids = {}
        for fmt in ("{label}", " {label}"):
            label_ids = {}
            ok = True
            for label in built.labels:
                full = tokenizer(
                    prompt + fmt.format(label=label),
                    add_special_tokens=add_special,
                ).input_ids
                added = full[len(prompt_ids):]
                if full[: len(prompt_ids)] == prompt_ids and len(added) == 1:
                    label_ids[label] = added[0]
                else:
                    ok = False
            if ok:
                break
        scores = [float(logits[label_ids[label]]) for label in built.labels]
        probs = softmax(scores)
        choice = built.keys[max(range(len(probs)), key=lambda i: probs[i])]
        # Decode-free: zero generated tokens.
        return {"preview": choice, "generated_tokens": 0, "prompt_tokens": len(prompt_ids)}

    def one_token_generation() -> dict:
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_tensor,
                attention_mask=torch.ones_like(input_tensor),
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = int(out.shape[1] - input_tensor.shape[1])
        return {
            "preview": tokenizer.decode(
                out[0, input_tensor.shape[1]:], skip_special_tokens=True
            ),
            "generated_tokens": generated,
            "prompt_tokens": len(prompt_ids),
        }

    def chat_think() -> dict:
        with torch.inference_mode():
            out = model.generate(
                input_ids=think_tensor,
                attention_mask=torch.ones_like(think_tensor),
                max_new_tokens=think_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = int(out.shape[1] - think_tensor.shape[1])
        text = tokenizer.decode(
            out[0, think_tensor.shape[1]:], skip_special_tokens=True
        )
        preview = _parse_think_answer(text, built.keys)
        return {"preview": preview, "generated_tokens": generated, "prompt_tokens": len(think_ids)}

    def chat_normal() -> dict:
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_tensor,
                attention_mask=torch.ones_like(input_tensor),
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = int(out.shape[1] - input_tensor.shape[1])
        return {
            "preview": tokenizer.decode(
                out[0, input_tensor.shape[1]:], skip_special_tokens=True
            ),
            "generated_tokens": generated,
            "prompt_tokens": len(prompt_ids),
        }

    print(f"warming up with {runs} runs each (normal/think caps: {gen_tokens}/{think_tokens} tokens) ...")
    report = {
        "model": model_id,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "input_tokens": len(prompt_ids),
        "think_tokens_limit": think_tokens,
        "normal_tokens_limit": gen_tokens,
        "runs": runs,
        "memory": _memory_mb(),
        "direct_logits": measure(direct_logits, runs),
        "chat_normal": measure(chat_normal, runs),
        "chat_think": measure(chat_think, runs),
        # Kept in raw JSON for SPEC.md §13.9; excluded from the HTML report.
        "one_token_generation": measure(one_token_generation, runs),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--think-tokens", type=int, default=256)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--output", default=None, help="write JSON report here")
    args = parser.parse_args()

    report = run(
        args.model, args.runs, args.gen_tokens, args.think_tokens,
        not args.no_chat_template,
    )

    print()
    header = f"{'mode':<24}{'cold ms':>10}{'warm p50':>10}{'warm p95':>10}{'prompt':>8}{'gen':>6}"
    print(header)
    print("-" * len(header))
    for mode in (
        "direct_logits",
        "chat_normal",
        "chat_think",
        "one_token_generation",
    ):
        entry = report[mode]
        print(
            f"{mode:<24}"
            f"{entry['cold_ms']:>10}"
            f"{str(entry['warm_p50_ms']):>10}"
            f"{str(entry['warm_p95_ms']):>10}"
            f"{entry['prompt_tokens']:>8}"
            f"{entry['generated_tokens']:>6}"
        )
    print()
    print(f"input tokens: {report['input_tokens']}")
    print(f"memory: {report['memory']}")
    print(
        "note: probabilities from direct_logits are relative label scores, "
        "not calibrated confidence"
    )

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nreport written to {args.output}")


if __name__ == "__main__":
    main()
