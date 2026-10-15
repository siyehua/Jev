# Jev — Logits-Based Decision API

A small HTTP service that keeps chat/generation separate from a dedicated
decision interface: `POST /v1/decision` evaluates a finite set of candidate
labels via **one forward pass**, reads the next-token logits, and returns a
structured JSON result — **without calling `model.generate()`** and without
decoding any output tokens.

Implements [SPEC.md](SPEC.md) (v0.1, MVP).

- **Backend:** Hugging Face Transformers causal LM (default: `Qwen/Qwen2.5-0.5B`)
- **Strategy:** `label_logits` — single-prompt label-logit scoring
- **Framework:** FastAPI + Uvicorn

> **Interpretation limit (SPEC.md §7.6):** the returned `probabilities` are
> softmax-normalized **relative label scores** over the labels in one request.
> They are **not** calibrated probabilities of real-world correctness, and they
> are not equivalent to asking the model to reason independently about each
> candidate (that would be a future `binary_logits` strategy).

## Quick start

```bash
pip install -r requirements.txt

python run.py --host 127.0.0.1 --port 8000
# model loads once at startup; ready when you see "service ready"
```

```bash
curl -s localhost:8000/v1/decision -H 'content-type: application/json' -d '{
  "model": "Qwen/Qwen2.5-0.5B",
  "input": "My package arrived two weeks late, and my card was charged twice.",
  "instructions": "Choose the category that best matches the customer's main issue.",
  "criteria": {
    "shipping": "A delivery, shipment, or package-arrival problem.",
    "billing": "A payment, invoice, or duplicate-charge problem.",
    "returns": "A return, refund, or exchange request."
  },
  "scoring": {"strategy": "label_logits", "temperature": 1.0},
  "request_id": "demo-1"
}'
```

Response (values illustrative):

```json
{
  "id": "dec_01m4q337q5zzncf1agzn542rfj",
  "model": "Qwen/Qwen2.5-0.5B",
  "strategy": "label_logits",
  "choice": "shipping",
  "probabilities": {"shipping": 0.777, "billing": 0.173, "returns": 0.050},
  "scores": {"shipping": 14.44, "billing": 12.94, "returns": 11.69},
  "usage": {"prompt_tokens": 134, "generated_tokens": 0},
  "request_id": "demo-1",
  "backend": "transformers",
  "label_mode": "letters",
  "prompt_template_version": "v1",
  "latency": {"tokenize_ms": 0.94, "forward_ms": 205.05, "postprocess_ms": 0.02, "total_ms": 220.07}
}
```

`GET /healthz` reports service health and whether the model is loaded.

## How it works

1. **Validate** the request (types, 2–26 criteria, non-empty text, `temperature > 0`).
2. **Build a deterministic prompt** (`jev/prompt.py`, template version `v1`):
   instructions and context are wrapped in explicit `<instructions>`/`<context>`
   delimiters; candidate labels are listed one per line. The model's chat
   template is applied when the tokenizer has one (`JEV_USE_CHAT_TEMPLATE=auto`).
3. **Resolve label tokens safely** (`jev/tokens.py`): for each label, tokenize
   the exact prompt and the prompt + label in the exact continuation form; the
   label is only accepted if the continuation adds **exactly one token** and
   the prompt prefix tokenization is unchanged. Otherwise the request fails
   with `422 unsupported_label_tokenization`.
4. **One forward pass** (`jev/backends/transformers_backend.py`): the model and
   tokenizer are loaded once at startup (`model.eval()`, `torch.inference_mode()`
   for every pass); `score_next_tokens` reads `outputs.logits[0, -1, token_ids]`.
   `model.generate()` is never called in this mode.
5. **Normalize** (`jev/scoring.py`): numerically stable softmax with the
   request temperature; `choice` is the argmax with **first-label-wins** tie
   breaking in request order.

### Label modes

- `letters` (default): compact keys `A`…`Z` are used at the answer position and
  mapped back to the business `criteria` keys in the response. This is why
  criteria are limited to 2–26 entries.
- `criteria_keys`: the criteria keys themselves must each be single-token
  continuations, or the request fails with `422`.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `JEV_MODEL_ID` | `Qwen/Qwen2.5-0.5B` | Model to load at startup (server-side only; never taken from requests) |
| `JEV_ALLOWED_MODELS` | *(the model id)* | Comma-separated model names the server will answer for |
| `JEV_DEVICE` | `auto` | `auto` → CUDA if available else CPU; or `cpu`/`cuda` |
| `JEV_DTYPE` | `auto` | `auto` → CPU: float32; CUDA: bfloat16 if supported else float16 |
| `JEV_LABEL_MODE` | `letters` | `letters` or `criteria_keys` |
| `JEV_USE_CHAT_TEMPLATE` | `auto` | `auto`/`true`/`false` |
| `JEV_MAX_INPUT_CHARS` | `8000` | Max `input` length (413 when exceeded) |
| `JEV_MAX_INSTRUCTIONS_CHARS` | `4000` | Max `instructions` length |
| `JEV_MAX_CRITERIA_VALUE_CHARS` | `2000` | Max criteria description length |
| `JEV_MAX_PROMPT_TOKENS` | `4096` | Max rendered prompt tokens (413) |
| `JEV_MAX_BODY_BYTES` | `1048576` | Max request body size (413) |
| `JEV_REQUEST_TIMEOUT_SECONDS` | `120` | Per-request wall-clock timeout (500 on timeout) |
| `JEV_MAX_CONCURRENCY` | `1` | Concurrent forward passes (sequential by default, SPEC §10) |
| `JEV_API_KEY` | *(none)* | When set, `Authorization: Bearer <key>` is required on `/v1/decision` |
| `JEV_LOG_LEVEL` | `INFO` | Log level |
| `JEV_LOG_INPUTS` | `false` | Log raw inputs — trusted dev environments only |

## API errors (SPEC.md §9)

| HTTP | Code | Condition |
|---:|---|---|
| 400 | `invalid_request` | Missing/wrong-typed fields, empty input, `temperature <= 0`, unknown strategy |
| 401 | `unauthorized` | API key required but missing/invalid (only when `JEV_API_KEY` is set) |
| 404 | `model_not_found` | Requested model is not the configured/served model |
| 413 | `input_too_large` | Char/token/body limits exceeded |
| 422 | `unsupported_label_tokenization` | A label does not map to exactly one valid continuation token |
| 503 | `model_not_ready` | Model not loaded |
| 500 | `inference_error` | Unexpected backend failure (no stack traces leaked) |

## Architecture

```
HTTP layer (jev/app.py)           routes, auth, limits, error mapping, access logs
  └─ DecisionService (jev/service.py)   orchestration, latency accounting
       ├─ PromptBuilder (jev/prompt.py)      deterministic, versioned template
       ├─ label-token validation (jev/tokens.py)
       ├─ softmax/argmax (jev/scoring.py)
       └─ DecisionBackend (jev/backends/base.py)
            ├─ TransformersBackend   direct logits, no generate()
            └─ FakeBackend           char-level, for tests/offline dev
```

Decision logic is fully decoupled from the model runtime behind the
`DecisionBackend` interface (`score_labels(prompt, labels)`), so alternative
backends can be added without touching application code.

## Ollama backend (one-token generation fallback)

The service can route decisions to a remote [Ollama](https://ollama.com)
server instead of loading a local Transformers model:

```bash
JEV_BACKEND=ollama \
JEV_MODEL_ID=qwen2.5:0.5b \
JEV_OLLAMA_BASE_URL=http://ollama-host:11434 \
python run.py
```

**Important semantics (SPEC.md §8).** The public Ollama API does not expose
pre-generation full-vocabulary logits — its `logprobs` fields describe
*generated* tokens. The Ollama backend therefore uses the documented
fallback mode, which is **not decode-free**:

1. It sends the decision prompt with `num_predict: 1`, `temperature: 0`,
   and top-k logprobs enabled (`logprobs: true`, `top_logprobs: N`).
2. It scores every label from the first generated position's top-k
   logprobs. Softmax over log-probabilities equals softmax over the
   underlying logits restricted to the same labels (shift invariance), so
   the normalized `probabilities` are consistent with direct-logits
   scoring of the same distribution. The `scores` field contains log
   probabilities (negative values), not raw logits.
3. Responses report the execution honestly:
   `"strategy": "single_token_generation"` and
   `"usage": {"generated_tokens": 1, ...}` — even though the request asked
   for `label_logits` semantics.

Limitations:

- A label whose token is **not within the returned top-k logprobs** cannot
  be scored; the request fails with `422 unsupported_label_tokenization`.
  Increase `JEV_OLLAMA_TOP_LOGPROBS` (default 20) or use more compact
  labels.
- Requires an Ollama version with logprobs support (v0.6+ recommended:
  `logprobs: true` + `top_logprobs: N` request shape).
- True decode-free scoring over Ollama is not possible without changes to
  the serving runtime itself (SPEC.md §8); this backend never claims
  `label_logits`.

Field-tested behavior against Ollama v0.12.11 (handled automatically by
the backend):

- `top_logprobs` is hard-capped at **20** by the server; the request is
  clamped. With the v2 prompt template ("Respond with the letter of the
  best label only"), letter labels A–Z stay concentrated in the top-20;
  natural-language answer tokens are pushed out instead.
- With `num_predict: 1` the server **omits the logprobs array entirely**
  (a server quirk); the backend requests 2 tokens and uses only the
  first logprobs position. `usage.generated_tokens` reports the server's
  honest `eval_count` (2).
- Model compatibility: standard instruct/chat models (gemma3, qwen2.5,
  llama3, ...) work — the model must emit its answer at the first
  generated position. **Reasoning models such as `gpt-oss` (harmony
  format) do not work**: they begin with analysis prose ("We need to
  choose..."), so the first position never contains the answer. Use a
  standard chat model for this backend.

Configuration:

| Variable | Default | Meaning |
|---|---|---|
| `JEV_BACKEND` | `transformers` | `transformers` (direct logits) or `ollama` (generation fallback) |
| `JEV_MODEL_ID` | `Qwen/Qwen2.5-0.5B` | For Ollama: the model name on the server (e.g. `qwen2.5:0.5b`) |
| `JEV_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address (server-side config only, never from requests) |
| `JEV_OLLAMA_MODE` | `chat` | `chat` → `/api/chat` (server applies chat template); `generate` → `/api/generate` with `raw: true` |
| `JEV_OLLAMA_TOP_LOGPROBS` | `20` | Top-k logprobs requested at the answer position |

`GET /healthz` reports the Ollama server version, mode, and that the model
was found. The backend verifies connectivity and model presence at startup
and fails fast with a clear error otherwise.

## Observability

Every request logs a structured JSON line (`jev.access`): request/id, model,
backend, strategy, temperature, criteria count, prompt tokens, tokenization /
forward / post-process / total latency, and success/error code. Raw input text
is **not** logged unless `JEV_LOG_INPUTS=true`.

## Tests

```bash
python -m pytest                       # unit + API tests (FakeBackend, no model needed)
python -m pytest tests/test_integration.py   # real-model tests (skipped if model unavailable)
```

Integration tests verify with a real local causal LM (Qwen2.5-0.5B, CPU):
the choice is always among the criteria keys; probabilities sum to 1;
`generate()` is never called (spy); the model loads only at startup; the exact
prompt suffix and label token IDs (`A`→32, `B`→33, `C`→34 for Qwen2.5) and
their round-trip decode; chat-template application; multi-token criteria keys
rejected in `criteria_keys` mode.

## Benchmark

```bash
python benchmarks/bench.py --runs 20 --gen-tokens 32 --output results.json
```

Compares, on identical weights/prompt/hardware with cold and warm runs
separately: (1) direct label-logit scoring, (2) one-token label generation,
(3) short natural-language generation; reports p50/p95, input tokens, and
memory. Do not copy numbers between machines — run it on your own hardware.
(A sample run on this machine's GPU is in `benchmarks/results.json`; the
direct-logits cold call includes CUDA initialization.)

## Security notes

- All request text is untrusted: user content is delimited in the prompt, and
  the scorer can only ever return one of the supplied labels — injection
  cannot widen the output space.
- Model selection is server-side only; requests cannot load arbitrary models.
- Input size, body size, timeout, and concurrency limits are configurable; raw
  logits and stack traces are never exposed.
