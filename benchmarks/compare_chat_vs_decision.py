#!/usr/bin/env python3
"""Compare the Jev decision API against a direct Ollama chat call.

Both modes send the SAME rendered prompt (template v2) and classify the
same input, then the script reports, per mode:

  - result      (chosen label / free-form answer)
  - latency     (client wall time, plus server-side total for /v1/decision)
  - token usage (prompt tokens; generated tokens)

Modes:
  1. decision : POST {jev_url}/v1/decision  (one-token fallback + top-k logprobs)
  2. chat     : POST {ollama_url}/api/chat  (normal generation, num_predict=16)

Stdlib-only, proxies disabled (bypasses corporate http_proxy issues).

Usage (on the server):
    python3 benchmarks/compare_chat_vs_decision.py --runs 3
Usage (on your PC, with both ports forwarded):
    python3 compare_chat_vs_decision.py --runs 3 \
        --jev-url http://localhost:8000 --ollama-url http://localhost:11434
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

# Mirrors jev/prompt.py template v2 — kept inline so the script is
# self-contained on machines without the jev package.
TEMPLATE_V2 = """You are a classification engine. Select exactly one label from the available labels.
Do not explain your answer.

Task:
<instructions>
{instructions}
</instructions>

Context:
<context>
{input}
</context>

Candidate labels:
{candidate_descriptions}

Respond with the letter of the best label only:
Answer:"""

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DEFAULT_INPUT = (
    "My package arrived two weeks late, and my card was charged twice."
)
DEFAULT_INSTRUCTIONS = (
    "Choose the category that best matches the customer's main issue."
)
DEFAULT_CRITERIA = {
    "shipping": "A delivery, shipment, or package-arrival problem.",
    "billing": "A payment, invoice, or duplicate-charge problem.",
    "returns": "A return, refund, or exchange request.",
}

# no proxy for any host — avoid corporate http_proxy hijacking localhost
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get_json(url: str, timeout: float = 5.0) -> dict:
    with OPENER.open(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_prompt(instructions: str, input_text: str, criteria: dict) -> str:
    lines = []
    for index, (key, value) in enumerate(criteria.items()):
        letter = LETTERS[index]
        lines.append(f"{letter}. {key} — {value}")
    return TEMPLATE_V2.format(
        instructions=" ".join(instructions.split()),
        input=" ".join(input_text.split()),
        candidate_descriptions="\n".join(lines),
    )


def post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    with OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_decision(args, prompt: str) -> dict:
    payload = {
        "model": args.model,
        "input": args.input,
        "instructions": args.instructions,
        "criteria": args.criteria,
        "scoring": {"temperature": 1.0},
    }
    headers = {"authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    started = time.perf_counter()
    body = post_json(f"{args.jev_url}/v1/decision", payload, headers, args.timeout)
    wall_ms = (time.perf_counter() - started) * 1000
    return {
        "result": body["choice"],
        "detail": (
            f"choice={body['choice']} "
            f"p={max(body['probabilities'].values()):.6f} "
            f"strategy={body['strategy']}"
        ),
        "wall_ms": wall_ms,
        "server_ms": body["latency"]["total_ms"],
        "prompt_tokens": body["usage"]["prompt_tokens"],
        "generated_tokens": body["usage"]["generated_tokens"],
    }


def call_chat(args, prompt: str) -> dict:
    payload = {
        "model": args.model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "num_predict": 16},
    }
    started = time.perf_counter()
    body = post_json(
        f"{args.ollama_url}/api/chat", payload, {}, args.timeout
    )
    wall_ms = (time.perf_counter() - started) * 1000
    text = (body.get("message") or {}).get("content", "")
    label = parse_label(text, args.criteria)
    return {
        "result": label,
        "detail": f"raw_answer={text.strip()!r}",
        "wall_ms": wall_ms,
        "server_ms": None,
        "prompt_tokens": body.get("prompt_eval_count"),
        "generated_tokens": body.get("eval_count"),
    }


def parse_label(text: str, criteria: dict) -> str | None:
    """Map a free-form chat answer back to a criteria key."""
    keys = list(criteria.keys())
    # Direct keyword match (longest first to avoid prefix collisions).
    for key in sorted(keys, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", text, re.IGNORECASE):
            return key
    # Letter match (template asks for a letter).
    for index, key in enumerate(keys):
        if re.search(rf"\b{LETTERS[index]}\b", text):
            return key
    return None


def summarize(rows: list[dict], name: str) -> None:
    walls = [row["wall_ms"] for row in rows]
    prompt_tokens = [row["prompt_tokens"] for row in rows]
    gen_tokens = [row["generated_tokens"] for row in rows]
    print(f"  {name}:")
    print(f"    wall latency   mean={statistics.mean(walls):8.1f}ms  "
          f"min={min(walls):8.1f}ms  max={max(walls):8.1f}ms")
    print(f"    prompt tokens  {prompt_tokens}")
    print(f"    generated      {gen_tokens}")
    print(f"    results        {[row['result'] for row in rows]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jev-url", default="http://localhost:8000")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma3:12b")
    parser.add_argument("--api-key", default="jev-sk-123456789")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--json-out",
        default=None,
        help="write raw rows + summary to this JSON file (for reports)",
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    args = parser.parse_args()
    args.criteria = DEFAULT_CRITERIA

    prompt = build_prompt(args.instructions, args.input, args.criteria)
    try:
        jev_health = get_json(f"{args.jev_url}/healthz")
    except Exception:
        jev_health = None
    print(f"model={args.model} runs={args.runs}")
    print(f"prompt_chars={len(prompt)} (identical prompt for both modes)\n")

    decision_rows, chat_rows = [], []
    print(f"{'run':>4}  {'mode':<10}{'wall ms':>10}{'prompt':>8}{'gen':>5}  detail")
    print("-" * 100)
    for run in range(1, args.runs + 1):
        for name, call, sink in (
            ("decision", call_decision, decision_rows),
            ("chat", call_chat, chat_rows),
        ):
            row = call(args, prompt)
            row["run"] = run
            sink.append(row)
            server = (
                f" (server {row['server_ms']:.0f}ms)" if row["server_ms"] else ""
            )
            print(f"{run:>4}  {name:<10}{row['wall_ms']:>10.1f}"
                  f"{str(row['prompt_tokens']):>8}{str(row['generated_tokens']):>5}"
                  f"  {row['detail']}{server}")

    print("\n=== Summary ===")
    summarize(decision_rows, "/v1/decision (one-token fallback)")
    summarize(chat_rows, "/api/chat (free-form generation)")

    agree = sum(
        1
        for d, c in zip(decision_rows, chat_rows)
        if d["result"] is not None and d["result"] == c["result"]
    )
    print(
        f"\n  label agreement: {agree}/{args.runs} runs chose the same label"
    )
    d_mean = statistics.mean(r["wall_ms"] for r in decision_rows)
    c_mean = statistics.mean(r["wall_ms"] for r in chat_rows)
    print(
        f"  latency ratio  : chat/decision = {c_mean / d_mean:.2f}x "
        f"(decision {d_mean:.1f}ms vs chat {c_mean:.1f}ms)"
    )
    print(
        "\n  note: probabilities from /v1/decision are relative label scores, "
        "not calibrated confidence (SPEC.md §7.6)."
    )

    if args.json_out:
        payload = {
            "meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "runs": args.runs,
                "jev_url": args.jev_url,
                "ollama_url": args.ollama_url,
                "prompt_chars": len(prompt),
                "jev_health": jev_health,
                "input": args.input,
                "instructions": args.instructions,
                "criteria": args.criteria,
            },
            "decision_rows": decision_rows,
            "chat_rows": chat_rows,
            "summary": {
                "agreement": agree,
                "decision_mean_ms": d_mean,
                "chat_mean_ms": c_mean,
            },
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\n  JSON written to {args.json_out}")


if __name__ == "__main__":
    sys.exit(main())
