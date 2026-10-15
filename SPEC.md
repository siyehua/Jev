# SPEC.md — Logits-Based Decision API

**Status:** Draft / MVP specification  
**Version:** 0.1  
**Primary backend:** Hugging Face Transformers (causal language models)  
**Primary endpoint:** `POST /v1/decision`

## 1. Summary

Build a small HTTP service that keeps a conventional chat/generation interface separate from a dedicated decision interface. The decision endpoint evaluates a finite set of candidate labels and returns a structured result without asking the language model to generate a complete answer.

For the primary implementation, the service loads a compatible Hugging Face causal language model once at startup, runs a forward pass, reads the next-token logits, scores the requested labels, and constructs the JSON response in application code. It must not call `model.generate()` in direct-logits mode.

The MVP uses **single-prompt label-logit scoring**: one prompt presents the context, instructions, and all candidate labels; the model's next-token scores for those labels determine the result. This is a design choice for simplicity and low latency. It is not identical to LLM2Jev's independent yes/no scoring strategy, and results may differ.

## 2. Goals

- Provide a dedicated `POST /v1/decision` API.
- Reuse an existing compatible open-weight causal language model; do not train or modify model weights.
- Load the model and tokenizer once at service startup, not on every request.
- In direct-logits mode, score labels from the forward-pass logits without decoding generated output tokens.
- Return a machine-readable choice, relative label probabilities, and useful execution metadata.
- Keep decision logic separate from the model runtime so another backend can be added later.
- Validate model/tokenizer compatibility and fail clearly rather than silently returning misleading scores.

## 3. Non-goals for MVP

- Training or fine-tuning a model.
- Guaranteeing that returned scores are calibrated probabilities of real-world correctness.
- Supporting every model architecture in the Hugging Face Hub. MVP targets causal language models whose Transformers implementation returns next-token logits.
- Exposing raw model logits over the public HTTP API.
- Implementing a replacement for a full inference engine such as vLLM, SGLang, or Ollama.
- Claiming parity with the proprietary or official implementation of any other decision system.

## 4. Terminology

- **Logits:** unnormalized scores over the model vocabulary for the next token.
- **Label:** a candidate key such as `A`, `B`, or `billing`.
- **Label-logit scoring:** one model forward pass; compare the next-token logits for the candidate label tokens.
- **Binary candidate scoring (future/optional):** run a yes/no scoring prompt for each candidate independently, then combine the resulting candidate scores. This usually requires more than one prompt evaluation.
- **Prefill-only / decode-free:** execute the input forward pass and read scores without autoregressively generating output tokens.
- **Relative probability:** a softmax-normalized score across the labels included in one request. It is not automatically a calibrated probability of correctness.

## 5. High-level architecture

```text
Client
  |
  | POST /v1/decision
  v
HTTP API / request validation
  |
  v
Prompt builder + label/token validation
  |
  v
Decision service
  |
  v
Backend interface
  |
  +-- Transformers direct-logits backend (MVP)
  |
  +-- Optional future backends
  |     +-- SGLang direct-logits adapter
  |     +-- MLX direct-logits adapter
  |     +-- Ollama one-token generation fallback (not true decode-free scoring)
  v
Score normalization + response builder
  |
  v
JSON response
```

### Component responsibilities

1. **HTTP layer:** parse requests, apply size limits, return stable errors.
2. **Prompt builder:** render a deterministic prompt from request fields and a configured template. It uses regular code and templates; it does not call another LLM to write prompts.
3. **Tokenizer/label validator:** ensure each candidate label maps to a single next token in the exact prompt continuation context, or reject/fallback according to configuration.
4. **Backend:** run the model forward pass and return the logits needed for the decision.
5. **Scorer:** select candidate scores, apply temperature and softmax, and choose the winning label.
6. **Response builder:** return choice, relative probabilities, and metadata.

## 6. API contract

### 6.1 `POST /v1/decision`

Content type: `application/json`

Example request:

```json
{
  "model": "local-model",
  "input": "My package arrived two weeks late, and my card was charged twice.",
  "instructions": "Choose the category that best matches the customer's main issue.",
  "criteria": {
    "shipping": "A delivery, shipment, or package-arrival problem.",
    "billing": "A payment, invoice, or duplicate-charge problem.",
    "returns": "A return, refund, or exchange request."
  },
  "scoring": {
    "strategy": "label_logits",
    "temperature": 1.0
  }
}
```

#### Request fields

| Field | Type | Required | Rules |
|---|---|---:|---|
| `model` | string | Yes | Configured model ID/name. MVP may accept only the model loaded by the server. |
| `input` | string | Yes | Context to classify; must not be empty after trimming. |
| `instructions` | string | Yes | The decision question or task instructions. |
| `criteria` | object | Yes | Map of label keys to human-readable meanings. Must contain 2–26 entries. Keys must be non-empty and unique. |
| `scoring.strategy` | string | No | Defaults to `label_logits`. MVP supports only `label_logits`. |
| `scoring.temperature` | number | No | Defaults to `1.0`; must be greater than `0`. Used only for the label-level softmax. |
| `request_id` | string | No | Caller-provided correlation ID, returned unchanged when provided. |

For the MVP, label keys must be usable as compact output labels and must each correspond to exactly one token in the exact continuation context. If a label does not meet this requirement, return a validation error rather than silently scoring the wrong token. The implementation should provide a clear error explaining that the selected label set is not supported by the loaded tokenizer/template.

#### Example response

```json
{
  "id": "dec_01JEXAMPLE",
  "model": "local-model",
  "strategy": "label_logits",
  "choice": "billing",
  "probabilities": {
    "shipping": 0.21,
    "billing": 0.68,
    "returns": 0.11
  },
  "scores": {
    "shipping": -1.42,
    "billing": -0.29,
    "returns": -2.06
  },
  "usage": {
    "prompt_tokens": 74,
    "generated_tokens": 0
  },
  "request_id": null
}
```

All numerical values above are illustrative. The `scores` are the selected token logits as used by the scorer, not calibrated probabilities. The response must explicitly report `generated_tokens: 0` only when the direct-logits backend did not decode output tokens.

### 6.2 `GET /healthz`

Returns service health and whether the model is loaded.

Example:

```json
{
  "status": "ok",
  "model_loaded": true,
  "backend": "transformers"
}
```

### 6.3 Existing chat API

If the application already has a normal chat endpoint, keep it unchanged. The decision API is a separate route, so route selection—not a `mode` field interpreted by the model—chooses the decision code path. This spec does not require the service to implement chat generation itself.

## 7. Decision algorithm: `label_logits`

### 7.1 Model lifecycle

- Load the tokenizer and model once during service startup.
- Call `model.eval()`.
- Use `torch.inference_mode()` for inference.
- Do not reload the model for each request.
- The implementation must choose and document its device/dtype policy (`auto`, CUDA, CPU, etc.).

### 7.2 Build the prompt

Use a deterministic template, for example:

```text
You are a classification engine. Select exactly one label from the available labels.
Do not explain your answer.

Task:
{instructions}

Context:
{input}

Candidate labels:
{candidate_descriptions}

Return the label only:
Answer:
```

The template is application code, not an LLM-generated prompt. The model's chat template should be applied where required by the selected model. Prompt templates should be configurable, versioned, and covered by tests.

### 7.3 Get label token IDs safely

Do not assume that `tokenizer.encode("A")` always represents the exact token that follows the prompt. Tokenization can depend on preceding whitespace and the prompt suffix.

For each label:

1. Tokenize the exact prompt prefix and the same prefix concatenated with the label in the precise continuation form used by the template.
2. Verify that the tokenized continuation adds exactly one token.
3. Record that token ID for this exact prompt/template context.
4. If the continuation adds zero or multiple tokens, reject the request with `422 unsupported_label_tokenization` in MVP.

The implementation may choose fixed compact label keys such as `A`, `B`, `C`, etc. The displayed business categories can remain descriptive values in `criteria`. Ensure the labels are distinguishable in the template and do not accidentally become part of the prompt before the answer position.

### 7.4 Forward pass and score extraction

Pseudocode:

```python
inputs = tokenizer(rendered_prompt, return_tensors="pt")
inputs = move_to_model_device(inputs)

with torch.inference_mode():
    outputs = model(**inputs)

next_token_logits = outputs.logits[0, -1, :]
label_logits = next_token_logits[label_token_ids]
probabilities = softmax(label_logits / temperature)
choice_index = argmax(probabilities)
```

This is conceptual pseudocode. Production code must handle model-specific input arguments, device maps, padding, and chat templates. Where supported, an implementation may use a model's last-logit optimization to avoid materializing logits for every prompt position, but correctness must not depend on that optimization.

### 7.5 Select and normalize

For candidate label \(i\), with selected next-token logit \(z_i\), compute:

\[
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
\]

where \(T > 0\) is the request temperature (default 1.0). Use a numerically stable softmax implementation.

- `choice` is the label with the highest probability.
- `probabilities` contains the normalized scores over the labels in this request only.
- Preserve request order for deterministic tie-breaking; the first label wins exact ties.
- Do not describe these scores as calibrated real-world confidence.

### 7.6 Important interpretation limit

Direct label-logit scoring measures how strongly the model prefers the next token corresponding to one label, given this prompt. It is not necessarily equivalent to asking the model to reason independently about each candidate. It can be sensitive to prompt wording, label tokenization, label order, and candidate descriptions.

For tasks where independent candidate judgments matter more than minimal latency, a future `binary_logits` strategy may run one yes/no prompt per candidate and then combine the scores. That strategy is slower and must be separately specified/tested; it is not part of the MVP.

## 8. Backend interface

Define a small internal contract similar to:

```python
class DecisionBackend:
    def load(self, config) -> None: ...
    def score_next_tokens(self, prompt: str, token_ids: list[int]) -> list[float]: ...
    def health(self) -> dict: ...
```

The exact Python types may change, but the interface must separate application logic from runtime-specific model execution.

### Transformers backend (MVP)

- Owns the tokenizer and model lifecycle.
- Calls the model's forward pass and returns selected next-token logits.
- Does not call `generate()` for direct-logits decisions.
- Must support the configured causal LM model; incompatible architectures fail during startup with a clear error.

### Ollama compatibility (future, optional)

There are two distinct modes and the code must not conflate them:

1. **One-token generation fallback:** request a compact label and limit generation to one token. This is simple but still performs generation/decode. It does not meet the direct-logits MVP requirement.
2. **True direct-logits mode:** requires an underlying supported interface that exposes the needed pre-generation logits or a change to the serving/runtime implementation. Do not assume that the public Ollama logprobs fields expose the full raw vocabulary logits from the final prompt position; current public logprob fields describe generated-token log probabilities and optional top-token alternatives.

If a fallback is implemented, responses must identify the actual strategy, e.g. `strategy: "single_token_generation"`, and report generated-token usage honestly.

## 9. Error handling

Return JSON errors using a stable shape:

```json
{
  "error": {
    "code": "unsupported_label_tokenization",
    "message": "Label 'billing' does not tokenize as one next token for this model and prompt template. Use compact labels or configure a compatible template."
  }
}
```

Expected cases:

| HTTP status | Error code | Condition |
|---:|---|---|
| `400` | `invalid_request` | Missing fields, wrong types, empty input/instructions |
| `404` | `model_not_found` | Requested model is not configured |
| `413` | `input_too_large` | Input exceeds configured request/token limit |
| `422` | `unsupported_label_tokenization` | One or more labels do not map to one valid continuation token |
| `503` | `model_not_ready` | Model has not loaded or is unavailable |
| `500` | `inference_error` | Unexpected backend failure; do not leak stack traces to clients |

Input limits must be configurable. Log request ID, model name, latency, token count, and error code; do not log sensitive input text by default.

## 10. Performance requirements

- Model loading occurs at startup or an explicit model-switch event, never once per request.
- Direct-logits decisions must not call `generate()`.
- The service should record separate durations for tokenization, model forward pass, and post-processing.
- Report input token count. For direct-logits mode, `generated_tokens` must be zero.
- Start with sequential request execution if needed for correctness; add batching/concurrency only after measuring GPU/CPU memory behavior.
- Avoid claiming a speedup without a benchmark. Compare against a baseline that generates a short label and against a normal full-text response on the same model, hardware, prompt, and warm/cold-cache conditions.
- Use KV cache or shared-prefix reuse only as a separately measured optimization. The simple `model(**inputs)` MVP should not claim cross-request/prefix cache reuse unless implemented.

## 11. Security and robustness

- Treat `input`, `instructions`, and criteria descriptions as untrusted user input.
- The prompt should explicitly delimit context and candidate descriptions. Prompt injection cannot be fully solved by delimiters alone; the scorer must only return one of the supplied labels.
- Never execute code or tools based on model output.
- Do not accept arbitrary model paths from unauthenticated requests. Configure allowed model IDs server-side.
- Apply maximum input length, request timeout, and concurrency limits.
- Do not expose raw logits or internal stack traces by default.
- Consider authentication/rate limits if the service is network-accessible.

## 12. Observability

Log at least:

- request ID;
- configured model and backend;
- scoring strategy and temperature;
- input token count;
- tokenization, forward-pass, and total latency;
- success/failure and stable error code.

Do not log raw inputs unless explicitly enabled for a trusted development environment.

## 13. Acceptance criteria

The MVP is complete when all of the following are true:

1. `POST /v1/decision` accepts a valid request and returns one supplied label.
2. The response probabilities sum to approximately `1.0` within floating-point tolerance.
3. The endpoint does not call `model.generate()` in `label_logits` mode.
4. The tokenizer and model load once, not per request.
5. An unknown model, malformed request, too-large input, and unsupported label tokenization return documented errors.
6. The implementation has a deterministic tie-breaking rule.
7. Tests verify the exact prompt suffix and label token IDs for the configured model/tokenizer.
8. Tests confirm `generated_tokens` is zero in direct-logits mode and nonzero/accurately described in a generation fallback.
9. A benchmark script compares direct-logits decisions with one-token generation and normal text generation under controlled conditions.
10. Documentation makes clear that returned probabilities are relative label scores, not guaranteed calibrated probabilities.

## 14. Test plan

### Unit tests

- Request schema validation.
- Prompt construction and escaping/delimiting of user content.
- Label tokenization checks, including labels that split into multiple tokens.
- Stable softmax and temperature behavior.
- Tie-breaking based on input order.
- Error response schemas.

### Integration tests

- Run a small local causal LM supported by the configured Transformers version.
- Verify output label is always among `criteria` keys.
- Verify no `generate()` call occurs in direct-logits mode (mock or spy on the model API).
- Verify model loading only occurs during startup.
- Verify CPU execution; add GPU tests when GPU CI is available.
- Verify chat templates are applied for instruct/chat models.

### Benchmark tests

Record warm and cold latency separately for:

1. Direct label-logit scoring.
2. One-token generated label.
3. Short natural-language generation.

Use identical model weights, prompt content, hardware, and load conditions. Report input tokens, output tokens, p50/p95 latency, and memory use. Do not use synthetic performance numbers in documentation.

## 15. Suggested implementation milestones

1. Implement a Transformers-only prototype with one configured model.
2. Implement `/v1/decision`, request validation, prompt building, and label token checks.
3. Add direct logits scoring and a stable JSON response.
4. Add unit/integration tests and token/latency metrics.
5. Benchmark against one-token generation.
6. Add configurable model selection, batching, or alternative backends only if required.
7. Consider `binary_logits` independent-candidate scoring as a separate strategy only if evaluation shows a meaningful quality benefit.

## 16. References

- Hugging Face Transformers model outputs (`CausalLMOutput.logits`): https://huggingface.co/docs/transformers/main_classes/output
- Transformers model outputs source: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py
- Ollama API OpenAPI specification (generation fields and logprobs): https://github.com/ollama/ollama/blob/main/docs/openapi.yaml
- Ollama API types (`Logprobs` and `TopLogprobs` describe generated-token log probabilities): https://github.com/ollama/ollama/blob/main/api/types.go

---

## Decision log

- **MVP backend:** Hugging Face Transformers. Rationale: direct access to causal-LM output logits in-process.
- **MVP scoring:** `label_logits`. Rationale: one prompt and one forward pass for a multi-class decision; low implementation complexity and latency.
- **Not included in MVP:** Ollama true direct-logits backend. Rationale: the public API's logprob fields are associated with generated tokens; a one-token generation fallback is a different execution mode.
- **Probability semantics:** report normalized relative scores, not calibrated confidence.
